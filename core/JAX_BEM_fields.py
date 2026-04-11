import jax.numpy as jnp
from jax import jit
from functools import partial
from core.JAX_BEM_mesh import get_triangle_quadrature
from core.JAX_BEM_config import COMPLEX_DTYPE

"""
Incident field functions for BEM.

Following bempp convention: RHS projections are computed as L2 inner products,
integrating the field against P1 basis functions over elements using quadrature.
For functions involving normals (like dp/dn), the ELEMENT normal is used at each
quadrature point, not vertex normals.
"""

#%% Point-wise field evaluation

@jit
def incident_field(x, k0, direction):
    """
    Plane wave incident field: p_inc = exp(i * k · x)

    Args:
        x: [N, 3] evaluation points
        k0: wavenumber
        direction: [3] incident wave direction (will be normalized)

    Returns:
        p_inc: [N] complex incident field values
    """
    direction = direction / jnp.linalg.norm(direction)
    k_dot_x = k0 * jnp.dot(x, direction)
    return jnp.exp(1j * k_dot_x)


@jit
def incident_field_normal_derivative(x, n, k0, direction):
    """
    Normal derivative of plane wave: dp_inc/dn = i * k · n * exp(i * k · x)

    Args:
        x: [N, 3] evaluation points
        n: [N, 3] or [3] normal vectors at evaluation points
        k0: wavenumber
        direction: [3] incident wave direction (will be normalized)

    Returns:
        dp_inc_dn: [N] complex normal derivative values
    """
    direction = direction / jnp.linalg.norm(direction)
    k_dot_x = k0 * jnp.dot(x, direction)
    # Handle both [N, 3] and [3] normal shapes
    if n.ndim == 1:
        k_dot_n = k0 * jnp.dot(n, direction)
    else:
        k_dot_n = k0 * jnp.sum(n * direction, axis=-1)
    return 1j * k_dot_n * jnp.exp(1j * k_dot_x)


#%% L2 projection to compute coefficients (matching bempp)

@partial(jit, static_argnames=['quad_order'])
def compute_incident_field(vertices, normals, elements, k0, direction, quad_order=4):
    """
    Compute incident field L2 projections: projections[i] = integral of (f * phi_i) over mesh.

    These are the raw inner products used directly as RHS terms in the BEM system
    (not M^{-1}-weighted coefficients).

    Args:
        vertices: [N, 3] vertex positions
        normals: [F, 3] element normals
        elements: [F, 3] triangle indices
        k0: wavenumber
        direction: [3] incident direction
        quad_order: quadrature order

    Returns:
        projections: [N] complex L2 inner products at vertices
    """
    n_verts = vertices.shape[0]
    n_faces = elements.shape[0]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    n_quad = quad_weights.shape[0]

    # P1 basis values at quadrature points [3, Q]
    xi, eta = quad_points[0], quad_points[1]
    basis_vals = jnp.stack([1.0 - xi - eta, xi, eta], axis=0)

    # Compute integration elements = |cross(e1, e2)| = 2 * triangle_area
    # This matches bempp: sqrt(det(J^T @ J)) where J = [e1, e2]
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = jnp.cross(e1, e2)
    int_elem = jnp.linalg.norm(cross, axis=1)  # [F] - NOT divided by 2!

    # Compute global quadrature points for all elements [F, Q, 3]
    # x = v0 + xi * (v1 - v0) + eta * (v2 - v0)
    global_points = (v0[:, None, :] +
                     xi[None, :, None] * e1[:, None, :] +
                     eta[None, :, None] * e2[:, None, :])  # [F, Q, 3]

    # Evaluate incident field at all quadrature points [F, Q]
    global_points_flat = global_points.reshape(-1, 3)  # [F*Q, 3]
    f_vals_flat = incident_field(global_points_flat, k0, direction)  # [F*Q]
    f_vals = f_vals_flat.reshape(n_faces, n_quad)  # [F, Q]

    # Compute projections: proj[i] = sum over elements containing vertex i of
    #   integral of (f * phi_i) over element
    # For each element: contribution to vertex j = sum_q (f[q] * phi_j[q] * w[q]) * int_elem
    local_proj = jnp.einsum('fq,jq,q,f->fj', f_vals, basis_vals, quad_weights, int_elem)  # [F, 3]

    # Scatter to global projections
    projections = jnp.zeros(n_verts, dtype=COMPLEX_DTYPE)
    for i in range(3):
        projections = projections.at[elements[:, i]].add(local_proj[:, i])

    return projections


@partial(jit, static_argnames=['quad_order'])
def compute_source_neumann_projection(vertices, elements, source_mask, quad_order=4):
    """
    Compute L2 projection of unit normal velocity prescribed on source elements.

    For source elements (source_mask == True): dp/dn = 1 (unit normal velocity)
    For wall elements (source_mask == False):  dp/dn = 0

    Returns projections[i] = integral of phi_i dS over source elements.

    Args:
        vertices: [N, 3] vertex positions
        elements: [F, 3] triangle indices
        source_mask: [F] boolean array, True for source elements
        quad_order: quadrature order

    Returns:
        projections: [N] complex projections (real-valued, returned as complex64)
    """
    n_verts = vertices.shape[0]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    xi, eta = quad_points[0], quad_points[1]
    basis_vals = jnp.stack([1.0 - xi - eta, xi, eta], axis=0)  # [3, Q]

    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = jnp.cross(e1, e2)
    int_elem = jnp.linalg.norm(cross, axis=1)  # [F]

    # Zero out non-source elements
    weighted_int_elem = int_elem * source_mask.astype(jnp.float32)  # [F]

    # local_proj[f, j] = sum_q basis_j(q) * w_q * weighted_int_elem[f]
    local_proj = jnp.einsum('jq,q,f->fj', basis_vals, quad_weights, weighted_int_elem)  # [F, 3]

    projections = jnp.zeros(n_verts, dtype=COMPLEX_DTYPE)
    projections = projections.at[elements.ravel()].add(
        local_proj.ravel()
    )

    return projections


@partial(jit, static_argnames=['quad_order'])
def compute_normal_derivative(vertices, normals, elements, k0, direction, quad_order=4):
    """
    Compute normal derivative L2 projections: projections[i] = integral of (dp/dn * phi_i) over mesh.

    Following bempp: dp/dn is evaluated at quadrature points using the ELEMENT normal
    (constant per element), not vertex normals. Returns raw inner products used directly
    as RHS terms in the BEM system.

    Args:
        vertices: [N, 3] vertex positions
        normals: [F, 3] element normals
        elements: [F, 3] triangle indices
        k0: wavenumber
        direction: [3] incident direction
        quad_order: quadrature order

    Returns:
        projections: [N] complex L2 inner products at vertices
    """
    n_verts = vertices.shape[0]
    #n_faces = elements.shape[0]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    #n_quad = quad_weights.shape[0]

    # P1 basis values at quadrature points [3, Q]
    xi, eta = quad_points[0], quad_points[1]
    basis_vals = jnp.stack([1.0 - xi - eta, xi, eta], axis=0)

    # Compute integration elements = |cross(e1, e2)| = 2 * triangle_area
    # This matches bempp: sqrt(det(J^T @ J)) where J = [e1, e2]
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = jnp.cross(e1, e2)
    int_elem = jnp.linalg.norm(cross, axis=1)  # [F] - NOT divided by 2!

    # Compute global quadrature points [F, Q, 3]
    global_points = (v0[:, None, :] +
                     xi[None, :, None] * e1[:, None, :] +
                     eta[None, :, None] * e2[:, None, :])

    # Evaluate normal derivative at quadrature points using ELEMENT normals
    # f(x, n) = i * k * (d · n) * exp(i * k * (d · x))
    direction_norm = direction / jnp.linalg.norm(direction)

    # k · x for all points [F, Q]
    k_dot_x = k0 * jnp.einsum('fqi,i->fq', global_points, direction_norm)

    # k · n using element normal (constant per element) [F]
    k_dot_n = k0 * jnp.dot(normals, direction_norm)  # [F]

    # Normal derivative values [F, Q]
    f_vals = 1j * k_dot_n[:, None] * jnp.exp(1j * k_dot_x)

    # Compute projections
    local_proj = jnp.einsum('fq,jq,q,f->fj', f_vals, basis_vals, quad_weights, int_elem)  # [F, 3]

    # Scatter to global projections
    projections = jnp.zeros(n_verts, dtype=COMPLEX_DTYPE)
    for i in range(3):
        projections = projections.at[elements[:, i]].add(local_proj[:, i])

    return projections
