import jax.numpy as jnp
import jax.lax as lax
from jax import jit, checkpoint
from functools import partial
from core.JAX_BEM_operators import get_active_reflections, reflect_points
from core.JAX_BEM_config import COMPLEX_DTYPE

M_INV_4PI = 1.0 / (4.0 * jnp.pi)

"""
Kirchhoff-Helmholtz propagation for JAX-BEM.

Propagates boundary solution to domain points using the double-layer potential:
    u(x) = ∫_Γ ∂G/∂n_y(x, y) u(y) dS(y)
"""

def create_domain_grid(grid_size, resolution, grid_center=None):
    """Create [resolution^3, 3] grid of points spanning grid_size in each dimension.

    Args:
        grid_size: side length of the cubic grid
        resolution: number of points per axis
        grid_center: [3] centre of the grid (default: origin)

    Returns:
        [resolution^3, 3] array of 3D points
    """
    x = jnp.linspace(-grid_size / 2, grid_size / 2, resolution)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing='ij')
    points = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    if grid_center is not None:
        points = points + jnp.asarray(grid_center)
    return points


@jit
def get_boundary_data(vertices, elements):
    """
    Extract element centroids, unit normals, and areas from mesh.

    Returns: (centroids [F,3], normals [F,3], areas [F])
    """
    v0, v1, v2 = vertices[elements[:, 0]], vertices[elements[:, 1]], vertices[elements[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = jnp.cross(v1 - v0, v2 - v0)
    norms = jnp.linalg.norm(cross, axis=1)
    return centroids, cross / norms[:, None], 0.5 * norms


def propagate(vertices, elements, node_values, k0, grid_size,
              resolution, symmetry=None, chunk_size=65536, grid_center=None,
              neumann_elem_values=None, space='P1'):
    """
    Matrix-free propagation using chunked computation.

    Computes the Kirchhoff-Helmholtz exterior representation:
        p(x) = K[p](x) - V[g](x)
    where K is the double-layer potential and V is the single-layer potential.

    Memory usage is O(chunk_size * F) instead of O(M * F).

    Args:
        vertices:           [N, 3] vertex positions
        elements:           [F, 3] triangle connectivity
        node_values:        solution DOF values — [N] for P1 (vertices), [F] for DP0 (elements)
        k0:                 wavenumber
        grid_size:          domain grid extent (cubic side length)
        resolution:         grid resolution (M = resolution^3)
        symmetry:           tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        chunk_size:         domain points per chunk (tune for memory/speed)
        grid_center:        [3] centre of domain grid (default: origin)
        neumann_elem_values: [F] element-wise Neumann data dp/dn; None → DL only
        space:              'P1' (default) or 'DP0'

    Returns:
        [resolution, resolution, resolution] domain pressure field
    """
    if symmetry is None:
        symmetry_tuple = (False, False, False)
    else:
        symmetry_tuple = tuple(bool(s) for s in symmetry)

    center = jnp.zeros(3) if grid_center is None else jnp.asarray(grid_center)

    n_faces = elements.shape[0]
    if neumann_elem_values is None:
        neumann_elem_values = jnp.zeros(n_faces, dtype=COMPLEX_DTYPE)
    else:
        neumann_elem_values = jnp.asarray(neumann_elem_values, dtype=COMPLEX_DTYPE)

    return _propagate_jit(vertices, elements, node_values, k0, grid_size,
                          resolution, chunk_size, symmetry_tuple, center,
                          neumann_elem_values, space)


@partial(jit, static_argnames=['resolution', 'chunk_size', 'symmetry', 'space'])
def _propagate_jit(vertices, elements, node_values, k0, grid_size,
                   resolution, chunk_size, symmetry, grid_center,
                   neumann_elem_values, space):
    """JIT-compiled propagation with symmetry as static argument.

    Computes the Kirchhoff-Helmholtz exterior representation:
        p(x) = K[p](x) - V[g](x)
    where K is the double-layer (DL) and V is the single-layer (SL).
    neumann_elem_values holds g per element; zeros → DL only.
    """
    active_reflections = get_active_reflections(symmetry)

    # Interpolate to element centroids for the DL potential kernel
    if space == 'P1':
        element_values = (node_values[elements[:, 0]] +
                          node_values[elements[:, 1]] +
                          node_values[elements[:, 2]]) / 3.0
    else:  # DP0: solution already lives at element centres
        element_values = node_values

    boundary_points, boundary_normals, boundary_areas = get_boundary_data(vertices, elements)

    domain_points = create_domain_grid(grid_size, resolution, grid_center)
    M = domain_points.shape[0]

    n_chunks = (M + chunk_size - 1) // chunk_size
    padded_size = n_chunks * chunk_size

    domain_padded = jnp.zeros((padded_size, 3))
    domain_padded = domain_padded.at[:M].set(domain_points)
    domain_chunked = domain_padded.reshape(n_chunks, chunk_size, 3)

    def compute_chunk_matvec(chunk_points):
        """K[p] - V[g] for a chunk of domain points."""
        r_diff = chunk_points[:, None, :] - boundary_points[None, :, :]
        R = jnp.linalg.norm(r_diff, axis=2)
        R_hat = r_diff / R[:, :, None]
        R_dot_n = jnp.sum(R_hat * boundary_normals[None, :, :], axis=2)

        G = jnp.exp(1j * k0 * R) * M_INV_4PI / R
        dG_dn = -1j * k0 * G * R_dot_n * (1.0 - 1.0 / (1j * k0 * R))

        # DL and SL weighted by element area  [chunk_size, F]
        DL = dG_dn * boundary_areas[None, :]
        SL = G * boundary_areas[None, :]
        result = DL @ element_values - SL @ neumann_elem_values

        for reflection in active_reflections:
            reflected_boundary_points = reflect_points(boundary_points, reflection)
            reflected_boundary_normals = reflect_points(boundary_normals, reflection)

            r_diff_img = chunk_points[:, None, :] - reflected_boundary_points[None, :, :]
            R_img = jnp.linalg.norm(r_diff_img, axis=2)
            R_hat_img = r_diff_img / R_img[:, :, None]
            R_dot_n_img = jnp.sum(R_hat_img * reflected_boundary_normals[None, :, :], axis=2)

            G_img = jnp.exp(1j * k0 * R_img) * M_INV_4PI / R_img
            dG_dn_img = -1j * k0 * G_img * R_dot_n_img * (1.0 - 1.0 / (1j * k0 * R_img))

            DL_img = dG_dn_img * boundary_areas[None, :]
            SL_img = G_img * boundary_areas[None, :]
            result = result + DL_img @ element_values - SL_img @ neumann_elem_values

        return result

    result_chunked = lax.map(checkpoint(compute_chunk_matvec), domain_chunked)

    domain_solution = result_chunked.reshape(-1)[:M]

    return domain_solution.reshape(resolution, resolution, resolution)


def propagate_to_points(vertices, elements, node_values, k0, eval_points,
                        symmetry=None, neumann_elem_values=None, space='P1'):
    """
    Evaluate Kirchhoff-Helmholtz exterior representation at arbitrary 3D points:
        p(x) = K[p](x) - V[g](x)

    Lighter-weight than propagate() — no chunking needed for small point sets.

    Args:
        vertices:            [N, 3] vertex positions
        elements:            [F, 3] triangle connectivity
        node_values:         DOF values — [N] for P1, [F] for DP0
        k0:                  wavenumber
        eval_points:         [M, 3] evaluation points
        symmetry:            tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        neumann_elem_values: [F] element-wise Neumann data dp/dn; None → DL only
        space:               'P1' (default) or 'DP0'

    Returns:
        [M] complex pressure values
    """
    if symmetry is None:
        symmetry_tuple = (False, False, False)
    else:
        symmetry_tuple = tuple(bool(s) for s in symmetry)

    n_faces = elements.shape[0]
    if neumann_elem_values is None:
        neumann_elem_values = jnp.zeros(n_faces, dtype=COMPLEX_DTYPE)
    else:
        neumann_elem_values = jnp.asarray(neumann_elem_values, dtype=COMPLEX_DTYPE)

    return _propagate_to_points_jit(vertices, elements, node_values, k0,
                                    eval_points, symmetry_tuple,
                                    neumann_elem_values, space)


@partial(jit, static_argnames=['symmetry', 'space'])
def _propagate_to_points_jit(vertices, elements, node_values, k0,
                              eval_points, symmetry, neumann_elem_values, space):
    """JIT-compiled evaluation at arbitrary points.

    Computes K[p] - V[g]: double-layer minus single-layer (Kirchhoff-Helmholtz).
    """
    active_reflections = get_active_reflections(symmetry)

    if space == 'P1':
        element_values = (node_values[elements[:, 0]] +
                          node_values[elements[:, 1]] +
                          node_values[elements[:, 2]]) / 3.0
    else:  # DP0
        element_values = node_values

    boundary_points, boundary_normals, boundary_areas = get_boundary_data(vertices, elements)

    # Direct contribution [M, F]
    r_diff = eval_points[:, None, :] - boundary_points[None, :, :]
    R = jnp.linalg.norm(r_diff, axis=2)
    R_hat = r_diff / R[:, :, None]
    R_dot_n = jnp.sum(R_hat * boundary_normals[None, :, :], axis=2)
    G = jnp.exp(1j * k0 * R) * M_INV_4PI / R
    dG_dn = -1j * k0 * G * R_dot_n * (1.0 - 1.0 / (1j * k0 * R))

    DL = dG_dn * boundary_areas[None, :]  # [M, F]
    SL = G * boundary_areas[None, :]      # [M, F]
    result = DL @ element_values - SL @ neumann_elem_values

    # Image contributions
    for reflection in active_reflections:
        reflected_boundary_points = reflect_points(boundary_points, reflection)
        reflected_boundary_normals = reflect_points(boundary_normals, reflection)

        r_diff_img = eval_points[:, None, :] - reflected_boundary_points[None, :, :]
        R_img = jnp.linalg.norm(r_diff_img, axis=2)
        R_hat_img = r_diff_img / R_img[:, :, None]
        R_dot_n_img = jnp.sum(R_hat_img * reflected_boundary_normals[None, :, :], axis=2)
        G_img = jnp.exp(1j * k0 * R_img) * M_INV_4PI / R_img
        dG_dn_img = -1j * k0 * G_img * R_dot_n_img * (1.0 - 1.0 / (1j * k0 * R_img))

        DL_img = dG_dn_img * boundary_areas[None, :]
        SL_img = G_img * boundary_areas[None, :]
        result = result + DL_img @ element_values - SL_img @ neumann_elem_values

    return result  # [M]
