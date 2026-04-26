import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from core.JAX_BEM_singular import (compute_coincident_double_layer_matrix,
                              compute_coincident_adjoint_double_layer_matrix,
                              compute_coincident_hypersingular_matrix,
                              compute_coincident_single_layer_matrix,
                              compute_edge_adjacent_double_layer_matrix,
                              compute_edge_adjacent_adjoint_double_layer_matrix,
                              compute_edge_adjacent_hypersingular_matrix,
                              compute_edge_adjacent_single_layer_matrix,
                              compute_vertex_adjacent_double_layer_matrix,
                              compute_vertex_adjacent_adjoint_double_layer_matrix,
                              compute_vertex_adjacent_hypersingular_matrix,
                              compute_vertex_adjacent_single_layer_matrix,
                             )
from core.JAX_BEM_mesh import (compute_jacobians,
                            compute_integration_elements,
                            compute_surface_curls,
                            p1_basis_functions,
                            dp0_basis_functions,
                            get_triangle_quadrature,
                            compute_element_quadrature_points
                            )

from core.JAX_BEM_config import COMPLEX_DTYPE, FLOAT_DTYPE

M_INV_4PI = 1.0 / (4.0 * jnp.pi)


#%% Symmetry Reflection Helpers

@jit
def reflect_points(points, symmetry):
    """
    Reflect points across active symmetry planes.

    Args:
        points: [..., 3] array of points
        symmetry: [3] boolean array for [XY, XZ, YZ] planes

    Returns:
        reflected: [..., 3] reflected points

    Symmetry planes:
        - XY (symmetry[0]): z=0 plane, reflect z → -z
        - XZ (symmetry[1]): y=0 plane, reflect y → -y
        - YZ (symmetry[2]): x=0 plane, reflect x → -x
    """
    # Build a sign multiplier: 1.0 if no reflection, -1.0 if reflection
    # symmetry[0] (XY) affects z (index 2)
    # symmetry[1] (XZ) affects y (index 1)
    # symmetry[2] (YZ) affects x (index 0)
    signs = jnp.array([
        jnp.where(symmetry[2], -1.0, 1.0),  # x sign (YZ plane)
        jnp.where(symmetry[1], -1.0, 1.0),  # y sign (XZ plane)
        jnp.where(symmetry[0], -1.0, 1.0),  # z sign (XY plane)
    ])
    return points * signs


@jit
def reflect_normals(normals, symmetry):
    """
    Reflect normal vectors across active symmetry planes.

    For a Neumann (sound-hard) boundary condition on the symmetry plane,
    the normal component perpendicular to the plane flips sign while
    tangential components stay the same.

    Args:
        normals: [..., 3] array of normal vectors
        symmetry: [3] boolean array for [XY, XZ, YZ] planes

    Returns:
        reflected: [..., 3] reflected normals
    """
    # Normal reflection is same as point reflection for Neumann BC
    return reflect_points(normals, symmetry)


@jit
def reflect_curl(curl, symmetry):
    """
    Reflect surface curl vectors across active symmetry planes.

    The surface curl is a pseudovector, so it transforms as:
    curl' = det(R) * R @ curl

    For odd number of plane reflections (det(R)=-1): curl' = -R @ curl
    For even number of plane reflections (det(R)=+1): curl' = +R @ curl

    Args:
        curl: [3, 3] surface curl vectors (curl[i,:] is curl for basis i)
        symmetry: [3] boolean array for [XY, XZ, YZ] planes

    Returns:
        reflected: [3, 3] reflected curl vectors
    """
    # Build reflection matrix diagonal: R = diag(sx, sy, sz)
    # where s_i = -1 if reflecting across that plane, +1 otherwise
    signs = jnp.array([
        jnp.where(symmetry[2], -1.0, 1.0),  # x sign (YZ plane)
        jnp.where(symmetry[1], -1.0, 1.0),  # y sign (XZ plane)
        jnp.where(symmetry[0], -1.0, 1.0),  # z sign (XY plane)
    ])

    # det(R) = product of signs = (-1)^(number of active planes)
    # For odd number of planes: det = -1, so curl' = -R @ curl
    # For even number of planes: det = +1, so curl' = +R @ curl
    det_R = signs[0] * signs[1] * signs[2]

    # curl' = det(R) * R @ curl = det(R) * signs * curl
    return det_R * curl * signs


def get_active_reflections(symmetry_tuple):
    """
    Generate all active reflection combinations for the given symmetry planes.

    This is a pure Python function called at trace time. For N active symmetry
    planes, returns 2^N - 1 reflection combinations (excluding the identity).

    Args:
        symmetry_tuple: tuple of 3 bools for (XY, XZ, YZ) symmetry planes

    Returns:
        List of jnp arrays, each [3] bool representing a reflection combination.
        Empty list if no symmetry planes are active.

    Example:
        symmetry_tuple = (False, True, True)  # XZ and YZ planes active
        Returns reflections for: [0,1,0], [0,0,1], [0,1,1]
    """
    sym_xy, sym_xz, sym_yz = symmetry_tuple
    reflections = []

    # Single plane reflections
    if sym_xy:
        reflections.append(jnp.array([True, False, False]))
    if sym_xz:
        reflections.append(jnp.array([False, True, False]))
    if sym_yz:
        reflections.append(jnp.array([False, False, True]))

    # Two plane reflections
    if sym_xy and sym_xz:
        reflections.append(jnp.array([True, True, False]))
    if sym_xz and sym_yz:
        reflections.append(jnp.array([False, True, True]))
    if sym_xy and sym_yz:
        reflections.append(jnp.array([True, False, True]))

    # Three plane reflection
    if sym_xy and sym_xz and sym_yz:
        reflections.append(jnp.array([True, True, True]))

    return reflections


#%% Mass Matrix Assembly (Identity Operator Weak Form)

@partial(jit, static_argnames=['quad_order', 'space'])
def assemble_mass_matrix(vertices, faces, quad_order=4, space='P1'):
    """
    Assemble the mass matrix (weak form of the identity operator).

    M_ij = ∫ φ_i φ_j dS

    Args:
        vertices:   [N, 3] vertex positions
        faces:      [F, 3] triangle connectivity
        quad_order: quadrature order (1, 3, 4, or 7). Default 4 matches bempp.
        space:      'P1' (default) or 'DP0'

    Returns:
        mass_matrix: [n_dofs, n_dofs] mass matrix (dense)
    """
    n_faces = faces.shape[0]
    n_local = 3 if space == 'P1' else 1
    n_dofs  = vertices.shape[0] if space == 'P1' else n_faces

    if space == 'P1':
        local2global = faces                                              # [F, 3]
        basis_vals = p1_basis_functions(get_triangle_quadrature(quad_order)[0])
    else:
        local2global = jnp.arange(n_faces, dtype=jnp.int32)[:, None]    # [F, 1]
        basis_vals = dp0_basis_functions(get_triangle_quadrature(quad_order)[0])

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    jacobians = compute_jacobians(vertices, faces)
    integration_elements = compute_integration_elements(jacobians)

    def compute_local_mass(elem_idx):
        int_elem = integration_elements[elem_idx]
        weighted_basis = basis_vals * (quad_weights * int_elem)
        return basis_vals @ weighted_basis.T  # [n_local, n_local]

    all_local = vmap(compute_local_mass)(jnp.arange(n_faces))  # [F, n_local, n_local]

    mass = jnp.zeros((n_dofs, n_dofs), dtype=FLOAT_DTYPE)
    test_idx  = jnp.broadcast_to(local2global[:, :, None], (n_faces, n_local, n_local)).ravel()
    trial_idx = jnp.broadcast_to(local2global[:, None, :], (n_faces, n_local, n_local)).ravel()
    mass = mass.at[test_idx, trial_idx].add(all_local.ravel())
    return mass

#%% Kernel Functions


@jit
def helmholtz_single_layer_kernel(test_points, trial_points, k0):
    """
    Evaluate Helmholtz single layer kernel for regular integration (JAX vectorized).

    Only called for non-adjacent element pairs, so no singularity handling needed.
    """
    diff = trial_points - test_points[:, None]  # [Q_test, Q_trial, 3]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # [Q_test, Q_trial]
    phase = jnp.exp(1j * k0 * dist)
    return phase * M_INV_4PI / dist

@jit
def helmholtz_double_layer_kernel(test_points, trial_points, trial_normals, k0):
    """
    Evaluate Helmholtz double layer for regular kernels (JAX vectorized).

    Only called for non-adjacent element pairs, so no singularity handling needed.
    """
    diff = trial_points - test_points[:, None]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    # Laplace gradient kernel
    laplace_grad = jnp.sum(diff * trial_normals, axis=-1) * M_INV_4PI / (dist ** 3)
    # Complex exponential
    phase = jnp.exp(1j * k0 * dist)
    # Double layer kernel: derivative brings down (i*k - 1/r) factor
    return laplace_grad * phase * (-1 + 1j * k0 * dist)


@jit
def helmholtz_adjoint_double_layer_kernel(test_points, trial_points, test_normals, k0):
    """
    Evaluate Helmholtz adjoint double layer for regular kernels (JAX vectorized).

    K'(x,y) = ∂G(x,y)/∂n(x) = (x-y)·n_x / (4π|x-y|³) * exp(ik|x-y|) * (-1 + ik|x-y|)

    Identical formula to the double layer but with diff = x-y (negated) and
    test normals n_x instead of trial normals n_y.

    Args:
        test_points:  [Q_test, 3]
        trial_points: [Q_trial, 3]
        test_normals: [Q_test, 3] — normals at test quadrature points
        k0: wavenumber
    Returns:
        kernel: [Q_test, Q_trial]
    """
    diff = test_points[:, None] - trial_points[None, :]   # x - y, [Q_test, Q_trial, 3]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    laplace_grad = jnp.sum(diff * test_normals[:, None, :], axis=-1) * M_INV_4PI / (dist ** 3)
    phase = jnp.exp(1j * k0 * dist)
    return laplace_grad * phase * (-1 + 1j * k0 * dist)


#%% Element Pair Local Matrix (for vmap)

@jit
def _compute_double_layer_contribution(test_points, trial_points, trial_normals,
                                        test_int_elem, trial_int_elem, k0,
                                        basis_vals, quad_weights):
    """Compute double layer contribution for one element pair (no symmetry)."""
    kernel_matrix = helmholtz_double_layer_kernel(test_points, trial_points, trial_normals, k0)
    weights_2d = (quad_weights[:, None] * quad_weights[None, :] *
                  test_int_elem * trial_int_elem)
    weighted_kernel = kernel_matrix * weights_2d
    return jnp.einsum('ip,jq,pq->ij', basis_vals, basis_vals, weighted_kernel)


def compute_double_layer_local_matrix(test_elem_data, trial_elem_data, k0,
                                      basis_vals, quad_weights, symmetry):
    """
    Compute local 3x3 matrix for one element pair.

    Following MATLAB Mesh2Mesh3D approach for method of images:
    - Direct: Elem2Elem3D(Ex, Ey, k, 'D')
    - For each active reflection combination: add Elem2Elem3D(Ex, Reflect(Ey, combo), k, 'D')

    With N symmetry planes, there are 2^N - 1 image contributions to add.

    Args:
        test_elem_data: dict with 'points' [Q,3], 'int_elem' scalar
        trial_elem_data: dict with 'points' [Q,3], 'normals' [Q,3], 'int_elem' scalar
        k0: wavenumber
        basis_vals: [3, Q] basis function values
        quad_weights: [Q] quadrature weights
        symmetry: tuple of 3 bools for (XY, XZ, YZ) symmetry planes (static)
    Returns:
        local_matrix: [3, 3]
    """
    test_points = test_elem_data['points']  # [Q, 3]
    trial_points = trial_elem_data['points']  # [Q, 3]
    trial_normals = trial_elem_data['normals']  # [Q, 3] or [3]
    test_int_elem = test_elem_data['int_elem']
    trial_int_elem = trial_elem_data['int_elem']

    # Direct contribution: Elem2Elem3D(Ex, Ey, k, 'D')
    local_matrix = _compute_double_layer_contribution(
        test_points, trial_points, trial_normals,
        test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)

    # Image contributions: loop over active reflection combinations
    # This loop unrolls at trace time since symmetry is static
    for reflection in get_active_reflections(symmetry):
        reflected_trial_points = reflect_points(trial_points, reflection)
        reflected_trial_normals = reflect_normals(trial_normals, reflection)
        local_matrix = local_matrix + _compute_double_layer_contribution(
            test_points, reflected_trial_points, reflected_trial_normals,
            test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)

    return local_matrix


@jit
def _compute_hypersingular_contribution(test_points, trial_points,
                                         test_normal, trial_normal,
                                         test_curl, trial_curl,
                                         test_int_elem, trial_int_elem, k0,
                                         basis_vals, quad_weights):
    """Compute hypersingular contribution for one element pair (no symmetry)."""
    # Single layer kernel
    kernel_matrix = helmholtz_single_layer_kernel(test_points, trial_points, k0)
    weights_2d = (quad_weights[:, None] * quad_weights[None, :] *
                  test_int_elem * trial_int_elem)
    weighted_kernel = kernel_matrix * weights_2d

    # Normal product
    normal_prod = jnp.dot(test_normal, trial_normal)

    # Curl product [3, 3]
    curl_product = test_curl @ trial_curl.T

    # Mass term
    mass_term = jnp.einsum('ip,jq,pq->ij', basis_vals, basis_vals, weighted_kernel)

    # Hypersingular formula
    kernel_sum = jnp.sum(weighted_kernel)
    return kernel_sum * curl_product - k0**2 * normal_prod * mass_term


def compute_hypersingular_local_matrix(test_elem_data, trial_elem_data, k0,
                                       basis_vals, quad_weights, symmetry):
    """
    Compute local 3x3 hypersingular matrix for one element pair.

    Following MATLAB Mesh2Mesh3D approach for method of images:
    - Direct: Elem2Elem3D(Ex, Ey, k, 'H')
    - For each active reflection combination: add Elem2Elem3D(Ex, Reflect(Ey, combo), k, 'H')

    With N symmetry planes, there are 2^N - 1 image contributions to add.

    Args:
        test_elem_data: dict with 'points', 'normals', 'curl', 'int_elem'
        trial_elem_data: dict with 'points', 'normals', 'curl', 'int_elem'
        k0: wavenumber
        basis_vals: [3, Q] basis function values
        quad_weights: [Q] quadrature weights
        symmetry: tuple of 3 bools for (XY, XZ, YZ) symmetry planes (static)
    Returns:
        local_matrix: [3, 3]
    """
    test_points = test_elem_data['points']
    test_normal = test_elem_data['normals']
    test_curl = test_elem_data['curl']
    test_int_elem = test_elem_data['int_elem']

    trial_points = trial_elem_data['points']
    trial_normal = trial_elem_data['normals']
    trial_curl = trial_elem_data['curl']
    trial_int_elem = trial_elem_data['int_elem']

    # Direct contribution: Elem2Elem3D(Ex, Ey, k, 'H')
    local_matrix = _compute_hypersingular_contribution(
        test_points, trial_points, test_normal, trial_normal,
        test_curl, trial_curl, test_int_elem, trial_int_elem, k0,
        basis_vals, quad_weights)

    # Image contributions: loop over active reflection combinations
    # This loop unrolls at trace time since symmetry is static
    for reflection in get_active_reflections(symmetry):
        reflected_trial_points = reflect_points(trial_points, reflection)
        reflected_trial_normal = reflect_normals(trial_normal, reflection)
        reflected_trial_curl = reflect_curl(trial_curl, reflection)
        local_matrix = local_matrix + _compute_hypersingular_contribution(
            test_points, reflected_trial_points, test_normal, reflected_trial_normal,
            test_curl, reflected_trial_curl, test_int_elem, trial_int_elem, k0,
            basis_vals, quad_weights)

    return local_matrix


@jit
def _compute_adjoint_double_layer_contribution(test_points, trial_points, test_normals,
                                                test_int_elem, trial_int_elem, k0,
                                                basis_vals, quad_weights):
    """Compute adjoint double layer contribution for one element pair (no symmetry)."""
    kernel_matrix = helmholtz_adjoint_double_layer_kernel(test_points, trial_points, test_normals, k0)
    weights_2d = (quad_weights[:, None] * quad_weights[None, :] *
                  test_int_elem * trial_int_elem)
    weighted_kernel = kernel_matrix * weights_2d
    return jnp.einsum('ip,jq,pq->ij', basis_vals, basis_vals, weighted_kernel)


def compute_adjoint_double_layer_local_matrix(test_elem_data, trial_elem_data, k0,
                                               basis_vals, quad_weights, symmetry):
    """
    Compute local 3x3 adjoint double layer matrix for one element pair.

    Method of images: only trial points are reflected; test normals stay fixed,
    because K'(x,y) = ∂G/∂n(x) uses the normal at the test point x.

    Args:
        test_elem_data: dict with 'points' [Q,3], 'normals' [Q,3], 'int_elem' scalar
        trial_elem_data: dict with 'points' [Q,3], 'int_elem' scalar
        k0: wavenumber
        basis_vals: [3, Q]
        quad_weights: [Q]
        symmetry: tuple of 3 bools for (XY, XZ, YZ) planes (static)
    Returns:
        local_matrix: [3, 3]
    """
    test_points   = test_elem_data['points']
    test_normals  = test_elem_data['normals']   # [Q, 3]
    test_int_elem = test_elem_data['int_elem']
    trial_points   = trial_elem_data['points']
    trial_int_elem = trial_elem_data['int_elem']

    local_matrix = _compute_adjoint_double_layer_contribution(
        test_points, trial_points, test_normals,
        test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)

    for reflection in get_active_reflections(symmetry):
        reflected_trial_points = reflect_points(trial_points, reflection)
        local_matrix = local_matrix + _compute_adjoint_double_layer_contribution(
            test_points, reflected_trial_points, test_normals,
            test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)

    return local_matrix


#%% vmap Assembly

def assemble_double_layer(vertices, faces, normals, k0, adjacency_data,
                          quad_order=4, singular_order=4, symmetry=None, space='P1'):
    """
    Assemble double layer operator K.

    Args:
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        normals:        [F, 3] element normals
        k0:             wavenumber
        adjacency_data: pre-computed adjacency lists from compute_adjacency_lists()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        space:          'P1' (default) or 'DP0'
    """
    (regular_test_indices, regular_trial_indices,
     edge_test_indices, edge_trial_indices,
     vertex_test_indices, vertex_trial_indices,
     edge_shared_vertices, vertex_shared_vertices) = adjacency_data

    symmetry_tuple = (False, False, False) if symmetry is None else tuple(bool(s) for s in symmetry)

    return _assemble_double_layer_jit(
        vertices, faces, normals, k0,
        regular_test_indices, regular_trial_indices,
        edge_test_indices, edge_trial_indices,
        vertex_test_indices, vertex_trial_indices,
        edge_shared_vertices, vertex_shared_vertices,
        quad_order, singular_order, symmetry_tuple, space
    )


@partial(jit, static_argnames=['quad_order', 'singular_order', 'symmetry', 'space'])
def _assemble_double_layer_jit(vertices, faces, normals, k0,
                                regular_test_indices, regular_trial_indices,
                                edge_test_indices, edge_trial_indices,
                                vertex_test_indices, vertex_trial_indices,
                                edge_shared_vertices, vertex_shared_vertices,
                                quad_order, singular_order, symmetry, space):
    """JIT-compiled double layer assembly with pre-computed adjacency lists."""
    n_faces = faces.shape[0]
    n_local = 3 if space == 'P1' else 1
    n_dofs  = vertices.shape[0] if space == 'P1' else n_faces
    local2global = faces if space == 'P1' else jnp.arange(n_faces, dtype=jnp.int32)[:, None]

    n_regular_pairs = regular_test_indices.shape[0]
    n_edge_pairs    = edge_test_indices.shape[0]
    n_vertex_pairs  = vertex_test_indices.shape[0]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    basis_vals = p1_basis_functions(quad_points) if space == 'P1' else dp0_basis_functions(quad_points)

    jacobians = compute_jacobians(vertices, faces)
    integration_elements = compute_integration_elements(jacobians)
    phys_points = compute_element_quadrature_points(vertices, faces, jacobians, quad_points)
    normals_at_quad = jnp.tile(normals[:, None, :], (1, len(quad_weights), 1))  # [F, Q, 3]

    active_reflections = get_active_reflections(symmetry)

    # =========================================================================
    # STEP 1: Compute regular (far-field) contributions for non-adjacent pairs only
    # =========================================================================
    def compute_regular_pair(pair_idx):
        test_idx = regular_test_indices[pair_idx]
        trial_idx = regular_trial_indices[pair_idx]

        test_points = phys_points[test_idx]
        test_int_elem = integration_elements[test_idx]
        trial_points = phys_points[trial_idx]
        trial_normals_q = normals_at_quad[trial_idx]
        trial_int_elem = integration_elements[trial_idx]

        test_data = {'points': test_points, 'int_elem': test_int_elem}
        trial_data = {'points': trial_points, 'normals': trial_normals_q, 'int_elem': trial_int_elem}

        return compute_double_layer_local_matrix(test_data, trial_data, k0, basis_vals, quad_weights, symmetry)

    regular_matrices = vmap(compute_regular_pair)(jnp.arange(n_regular_pairs))  # [n_regular, 3, 3]

    # =========================================================================
    # STEP 2: Compute coincident singular contributions (diagonal: F pairs)
    # For singular pairs, direct contribution uses singular quadrature,
    # but image contribution uses regular quadrature (reflected element is far)
    # =========================================================================
    def compute_coincident(elem_idx):
        elem_verts  = vertices[faces[elem_idx]]
        elem_normal = normals[elem_idx]
        result = compute_coincident_double_layer_matrix(
            elem_verts, elem_normal, k0, order=singular_order, space=space)

        test_points    = phys_points[elem_idx]
        test_int_elem  = integration_elements[elem_idx]
        trial_normals_q = normals_at_quad[elem_idx]
        trial_int_elem = integration_elements[elem_idx]

        for reflection in active_reflections:
            reflected_trial_points  = reflect_points(phys_points[elem_idx], reflection)
            reflected_trial_normals = reflect_normals(trial_normals_q, reflection)
            result = result + _compute_double_layer_contribution(
                test_points, reflected_trial_points, reflected_trial_normals,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    coincident_matrices = vmap(compute_coincident)(jnp.arange(n_faces))

    # =========================================================================
    # STEP 3: Compute edge-adjacent singular contributions
    # =========================================================================
    def compute_edge_singular(pair_idx):
        test_idx  = edge_test_indices[pair_idx]
        trial_idx = edge_trial_indices[pair_idx]
        test_verts   = vertices[faces[test_idx]]
        trial_verts  = vertices[faces[trial_idx]]
        test_normal  = normals[test_idx]
        trial_normal = normals[trial_idx]
        tv1  = edge_shared_vertices[pair_idx, 0]
        tv2  = edge_shared_vertices[pair_idx, 1]
        trv1 = edge_shared_vertices[pair_idx, 2]
        trv2 = edge_shared_vertices[pair_idx, 3]

        result = compute_edge_adjacent_double_layer_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, tv2, trv1, trv2, order=singular_order, space=space)

        test_points    = phys_points[test_idx]
        test_int_elem  = integration_elements[test_idx]
        trial_normals_q = normals_at_quad[trial_idx]
        trial_int_elem = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points  = reflect_points(phys_points[trial_idx], reflection)
            reflected_trial_normals = reflect_normals(trial_normals_q, reflection)
            result = result + _compute_double_layer_contribution(
                test_points, reflected_trial_points, reflected_trial_normals,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    edge_matrices = vmap(compute_edge_singular)(jnp.arange(n_edge_pairs))

    # =========================================================================
    # STEP 4: Compute vertex-adjacent singular contributions
    # =========================================================================
    def compute_vertex_singular(pair_idx):
        test_idx  = vertex_test_indices[pair_idx]
        trial_idx = vertex_trial_indices[pair_idx]
        test_verts   = vertices[faces[test_idx]]
        trial_verts  = vertices[faces[trial_idx]]
        test_normal  = normals[test_idx]
        trial_normal = normals[trial_idx]
        tv1  = vertex_shared_vertices[pair_idx, 0]
        trv1 = vertex_shared_vertices[pair_idx, 1]

        result = compute_vertex_adjacent_double_layer_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, trv1, order=singular_order, space=space)

        test_points    = phys_points[test_idx]
        test_int_elem  = integration_elements[test_idx]
        trial_normals_q = normals_at_quad[trial_idx]
        trial_int_elem = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points  = reflect_points(phys_points[trial_idx], reflection)
            reflected_trial_normals = reflect_normals(trial_normals_q, reflection)
            result = result + _compute_double_layer_contribution(
                test_points, reflected_trial_points, reflected_trial_normals,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    vertex_matrices = vmap(compute_vertex_singular)(jnp.arange(n_vertex_pairs))

    # =========================================================================
    # STEP 5: Scatter all contributions to global matrix
    # =========================================================================
    operator = jnp.zeros((n_dofs, n_dofs), dtype=COMPLEX_DTYPE)

    reg_test_l2g  = local2global[regular_test_indices]   # [n_regular, n_local]
    reg_trial_l2g = local2global[regular_trial_indices]  # [n_regular, n_local]
    reg_test_idx  = jnp.broadcast_to(reg_test_l2g[:, :, None], (n_regular_pairs, n_local, n_local)).ravel()
    reg_trial_idx = jnp.broadcast_to(reg_trial_l2g[:, None, :], (n_regular_pairs, n_local, n_local)).ravel()
    operator = operator.at[reg_test_idx, reg_trial_idx].add(regular_matrices.ravel())

    diag_test_idx  = jnp.broadcast_to(local2global[:, :, None], (n_faces, n_local, n_local)).ravel()
    diag_trial_idx = jnp.broadcast_to(local2global[:, None, :], (n_faces, n_local, n_local)).ravel()
    operator = operator.at[diag_test_idx, diag_trial_idx].add(coincident_matrices.ravel())

    edge_test_l2g  = local2global[edge_test_indices]
    edge_trial_l2g = local2global[edge_trial_indices]
    edge_test_idx  = jnp.broadcast_to(edge_test_l2g[:, :, None], (n_edge_pairs, n_local, n_local)).ravel()
    edge_trial_idx = jnp.broadcast_to(edge_trial_l2g[:, None, :], (n_edge_pairs, n_local, n_local)).ravel()
    operator = operator.at[edge_test_idx, edge_trial_idx].add(edge_matrices.ravel())

    vert_test_l2g  = local2global[vertex_test_indices]
    vert_trial_l2g = local2global[vertex_trial_indices]
    vert_test_idx  = jnp.broadcast_to(vert_test_l2g[:, :, None], (n_vertex_pairs, n_local, n_local)).ravel()
    vert_trial_idx = jnp.broadcast_to(vert_trial_l2g[:, None, :], (n_vertex_pairs, n_local, n_local)).ravel()
    operator = operator.at[vert_test_idx, vert_trial_idx].add(vertex_matrices.ravel())

    return operator


def assemble_hypersingular(vertices, faces, normals, k0, adjacency_data,
                           quad_order=4, singular_order=4, symmetry=None, space='P1'):
    """
    Assemble hypersingular operator W.

    Args:
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        normals:        [F, 3] element normals
        k0:             wavenumber
        adjacency_data: pre-computed adjacency lists from compute_adjacency_lists()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        space:          'P1' (default) or 'DP0'
    """
    (regular_test_indices, regular_trial_indices,
     edge_test_indices, edge_trial_indices,
     vertex_test_indices, vertex_trial_indices,
     edge_shared_vertices, vertex_shared_vertices) = adjacency_data

    symmetry_tuple = (False, False, False) if symmetry is None else tuple(bool(s) for s in symmetry)

    return _assemble_hypersingular_jit(
        vertices, faces, normals, k0,
        regular_test_indices, regular_trial_indices,
        edge_test_indices, edge_trial_indices,
        vertex_test_indices, vertex_trial_indices,
        edge_shared_vertices, vertex_shared_vertices,
        quad_order, singular_order, symmetry_tuple, space
    )


@partial(jit, static_argnames=['quad_order', 'singular_order', 'symmetry', 'space'])
def _assemble_hypersingular_jit(vertices, faces, normals, k0,
                                 regular_test_indices, regular_trial_indices,
                                 edge_test_indices, edge_trial_indices,
                                 vertex_test_indices, vertex_trial_indices,
                                 edge_shared_vertices, vertex_shared_vertices,
                                 quad_order, singular_order, symmetry, space):
    """JIT-compiled hypersingular assembly with pre-computed adjacency lists."""
    n_faces = faces.shape[0]
    n_local = 3 if space == 'P1' else 1
    n_dofs  = vertices.shape[0] if space == 'P1' else n_faces
    local2global = faces if space == 'P1' else jnp.arange(n_faces, dtype=jnp.int32)[:, None]

    n_regular_pairs = regular_test_indices.shape[0]
    n_edge_pairs    = edge_test_indices.shape[0]
    n_vertex_pairs  = vertex_test_indices.shape[0]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    basis_vals = p1_basis_functions(quad_points) if space == 'P1' else dp0_basis_functions(quad_points)

    jacobians = compute_jacobians(vertices, faces)
    integration_elements = compute_integration_elements(jacobians)
    # Surface curls: [F, 3, 3] for P1; [F, 1, 3] zeros for DP0 (constant basis → zero curl)
    surface_curls = (compute_surface_curls(jacobians, normals)
                     if space == 'P1' else jnp.zeros((n_faces, 1, 3)))
    phys_points = compute_element_quadrature_points(vertices, faces, jacobians, quad_points)

    active_reflections = get_active_reflections(symmetry)

    # =========================================================================
    # STEP 1: Regular (far-field) contributions
    # =========================================================================
    def compute_regular_pair(pair_idx):
        test_idx  = regular_test_indices[pair_idx]
        trial_idx = regular_trial_indices[pair_idx]
        test_data  = {'points': phys_points[test_idx],  'normals': normals[test_idx],
                      'curl': surface_curls[test_idx],  'int_elem': integration_elements[test_idx]}
        trial_data = {'points': phys_points[trial_idx], 'normals': normals[trial_idx],
                      'curl': surface_curls[trial_idx], 'int_elem': integration_elements[trial_idx]}
        return compute_hypersingular_local_matrix(test_data, trial_data, k0, basis_vals, quad_weights, symmetry)

    regular_matrices = vmap(compute_regular_pair)(jnp.arange(n_regular_pairs))

    # =========================================================================
    # STEP 2: Coincident singular contributions
    # =========================================================================
    def compute_coincident(elem_idx):
        elem_verts  = vertices[faces[elem_idx]]
        elem_normal = normals[elem_idx]
        result = compute_coincident_hypersingular_matrix(
            elem_verts, elem_normal, k0, order=singular_order, space=space)

        test_points   = phys_points[elem_idx]
        test_curl     = surface_curls[elem_idx]
        test_int_elem = integration_elements[elem_idx]
        trial_curl    = surface_curls[elem_idx]
        trial_normal  = normals[elem_idx]
        trial_int_elem = integration_elements[elem_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[elem_idx], reflection)
            reflected_trial_normal = reflect_normals(trial_normal, reflection)
            reflected_trial_curl   = reflect_curl(trial_curl, reflection)
            result = result + _compute_hypersingular_contribution(
                test_points, reflected_trial_points, elem_normal, reflected_trial_normal,
                test_curl, reflected_trial_curl, test_int_elem, trial_int_elem, k0,
                basis_vals, quad_weights)
        return result

    coincident_matrices = vmap(compute_coincident)(jnp.arange(n_faces))

    # =========================================================================
    # STEP 3: Edge-adjacent singular contributions
    # =========================================================================
    def compute_edge_singular(pair_idx):
        test_idx  = edge_test_indices[pair_idx]
        trial_idx = edge_trial_indices[pair_idx]
        test_verts   = vertices[faces[test_idx]]
        trial_verts  = vertices[faces[trial_idx]]
        test_normal  = normals[test_idx]
        trial_normal = normals[trial_idx]
        tv1  = edge_shared_vertices[pair_idx, 0]
        tv2  = edge_shared_vertices[pair_idx, 1]
        trv1 = edge_shared_vertices[pair_idx, 2]
        trv2 = edge_shared_vertices[pair_idx, 3]

        result = compute_edge_adjacent_hypersingular_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, tv2, trv1, trv2, order=singular_order, space=space)

        test_points    = phys_points[test_idx]
        test_curl      = surface_curls[test_idx]
        test_int_elem  = integration_elements[test_idx]
        trial_curl     = surface_curls[trial_idx]
        trial_int_elem = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[trial_idx], reflection)
            reflected_trial_normal = reflect_normals(normals[trial_idx], reflection)
            reflected_trial_curl   = reflect_curl(trial_curl, reflection)
            result = result + _compute_hypersingular_contribution(
                test_points, reflected_trial_points, test_normal, reflected_trial_normal,
                test_curl, reflected_trial_curl, test_int_elem, trial_int_elem, k0,
                basis_vals, quad_weights)
        return result

    edge_matrices = vmap(compute_edge_singular)(jnp.arange(n_edge_pairs))

    # =========================================================================
    # STEP 4: Vertex-adjacent singular contributions
    # =========================================================================
    def compute_vertex_singular(pair_idx):
        test_idx  = vertex_test_indices[pair_idx]
        trial_idx = vertex_trial_indices[pair_idx]
        test_verts   = vertices[faces[test_idx]]
        trial_verts  = vertices[faces[trial_idx]]
        test_normal  = normals[test_idx]
        trial_normal = normals[trial_idx]
        tv1  = vertex_shared_vertices[pair_idx, 0]
        trv1 = vertex_shared_vertices[pair_idx, 1]

        result = compute_vertex_adjacent_hypersingular_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, trv1, order=singular_order, space=space)

        test_points    = phys_points[test_idx]
        test_curl      = surface_curls[test_idx]
        test_int_elem  = integration_elements[test_idx]
        trial_curl     = surface_curls[trial_idx]
        trial_int_elem = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[trial_idx], reflection)
            reflected_trial_normal = reflect_normals(normals[trial_idx], reflection)
            reflected_trial_curl   = reflect_curl(trial_curl, reflection)
            result = result + _compute_hypersingular_contribution(
                test_points, reflected_trial_points, test_normal, reflected_trial_normal,
                test_curl, reflected_trial_curl, test_int_elem, trial_int_elem, k0,
                basis_vals, quad_weights)
        return result

    vertex_matrices = vmap(compute_vertex_singular)(jnp.arange(n_vertex_pairs))

    # =========================================================================
    # STEP 5: Scatter
    # =========================================================================
    operator = jnp.zeros((n_dofs, n_dofs), dtype=COMPLEX_DTYPE)

    reg_test_l2g  = local2global[regular_test_indices]
    reg_trial_l2g = local2global[regular_trial_indices]
    operator = operator.at[
        jnp.broadcast_to(reg_test_l2g[:,  :, None], (n_regular_pairs, n_local, n_local)).ravel(),
        jnp.broadcast_to(reg_trial_l2g[:, None, :], (n_regular_pairs, n_local, n_local)).ravel(),
    ].add(regular_matrices.ravel())

    operator = operator.at[
        jnp.broadcast_to(local2global[:, :, None], (n_faces, n_local, n_local)).ravel(),
        jnp.broadcast_to(local2global[:, None, :], (n_faces, n_local, n_local)).ravel(),
    ].add(coincident_matrices.ravel())

    edge_test_l2g  = local2global[edge_test_indices]
    edge_trial_l2g = local2global[edge_trial_indices]
    operator = operator.at[
        jnp.broadcast_to(edge_test_l2g[:,  :, None], (n_edge_pairs, n_local, n_local)).ravel(),
        jnp.broadcast_to(edge_trial_l2g[:, None, :], (n_edge_pairs, n_local, n_local)).ravel(),
    ].add(edge_matrices.ravel())

    vert_test_l2g  = local2global[vertex_test_indices]
    vert_trial_l2g = local2global[vertex_trial_indices]
    operator = operator.at[
        jnp.broadcast_to(vert_test_l2g[:,  :, None], (n_vertex_pairs, n_local, n_local)).ravel(),
        jnp.broadcast_to(vert_trial_l2g[:, None, :], (n_vertex_pairs, n_local, n_local)).ravel(),
    ].add(vertex_matrices.ravel())

    return operator

#%% Adjoint Double Layer Assembly

def assemble_adjoint_double_layer(vertices, faces, normals, k0, adjacency_data,
                                   quad_order=4, singular_order=4, symmetry=None, space='P1'):
    """
    Assemble adjoint double layer operator K'.

    K'(x,y) = ∂G(x,y)/∂n(x) — normal derivative at the test point.

    Args:
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        normals:        [F, 3] element normals
        k0:             wavenumber
        adjacency_data: pre-computed adjacency lists from load_mesh()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        space:          'P1' (default) or 'DP0'
    """
    (regular_test_indices, regular_trial_indices,
     edge_test_indices, edge_trial_indices,
     vertex_test_indices, vertex_trial_indices,
     edge_shared_vertices, vertex_shared_vertices) = adjacency_data

    symmetry_tuple = (False, False, False) if symmetry is None else tuple(bool(s) for s in symmetry)

    return _assemble_adjoint_double_layer_jit(
        vertices, faces, normals, k0,
        regular_test_indices, regular_trial_indices,
        edge_test_indices, edge_trial_indices,
        vertex_test_indices, vertex_trial_indices,
        edge_shared_vertices, vertex_shared_vertices,
        quad_order, singular_order, symmetry_tuple, space
    )


@partial(jit, static_argnames=['quad_order', 'singular_order', 'symmetry', 'space'])
def _assemble_adjoint_double_layer_jit(vertices, faces, normals, k0,
                                        regular_test_indices, regular_trial_indices,
                                        edge_test_indices, edge_trial_indices,
                                        vertex_test_indices, vertex_trial_indices,
                                        edge_shared_vertices, vertex_shared_vertices,
                                        quad_order, singular_order, symmetry, space):
    """JIT-compiled adjoint double layer assembly."""
    n_faces = faces.shape[0]
    n_local = 3 if space == 'P1' else 1
    n_dofs  = vertices.shape[0] if space == 'P1' else n_faces
    local2global = faces if space == 'P1' else jnp.arange(n_faces, dtype=jnp.int32)[:, None]

    n_regular_pairs = regular_test_indices.shape[0]
    n_edge_pairs    = edge_test_indices.shape[0]
    n_vertex_pairs  = vertex_test_indices.shape[0]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    basis_vals = p1_basis_functions(quad_points) if space == 'P1' else dp0_basis_functions(quad_points)

    jacobians = compute_jacobians(vertices, faces)
    integration_elements = compute_integration_elements(jacobians)
    phys_points = compute_element_quadrature_points(vertices, faces, jacobians, quad_points)
    normals_at_quad = jnp.tile(normals[:, None, :], (1, len(quad_weights), 1))

    active_reflections = get_active_reflections(symmetry)

    # =========================================================================
    # STEP 1: Regular pairs — use test normals n_x (not trial normals)
    # =========================================================================
    def compute_regular_pair(pair_idx):
        test_idx  = regular_test_indices[pair_idx]
        trial_idx = regular_trial_indices[pair_idx]

        test_points    = phys_points[test_idx]
        test_normals_q = normals_at_quad[test_idx]   # K' uses test normals
        test_int_elem  = integration_elements[test_idx]
        trial_points   = phys_points[trial_idx]
        trial_int_elem = integration_elements[trial_idx]

        test_data  = {'points': test_points, 'normals': test_normals_q, 'int_elem': test_int_elem}
        trial_data = {'points': trial_points, 'int_elem': trial_int_elem}

        return compute_adjoint_double_layer_local_matrix(
            test_data, trial_data, k0, basis_vals, quad_weights, symmetry)

    regular_matrices = vmap(compute_regular_pair)(jnp.arange(n_regular_pairs))

    # =========================================================================
    # STEP 2: Coincident singular contributions
    # =========================================================================
    def compute_coincident(elem_idx):
        elem_verts  = vertices[faces[elem_idx]]
        elem_normal = normals[elem_idx]
        result = compute_coincident_adjoint_double_layer_matrix(
            elem_verts, elem_normal, k0, order=singular_order, space=space)

        test_points    = phys_points[elem_idx]
        test_normals_q = normals_at_quad[elem_idx]
        test_int_elem  = integration_elements[elem_idx]
        trial_int_elem = integration_elements[elem_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[elem_idx], reflection)
            result = result + _compute_adjoint_double_layer_contribution(
                test_points, reflected_trial_points, test_normals_q,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    coincident_matrices = vmap(compute_coincident)(jnp.arange(n_faces))

    # =========================================================================
    # STEP 3: Edge-adjacent singular contributions
    # =========================================================================
    def compute_edge_singular(pair_idx):
        test_idx  = edge_test_indices[pair_idx]
        trial_idx = edge_trial_indices[pair_idx]
        test_verts   = vertices[faces[test_idx]]
        trial_verts  = vertices[faces[trial_idx]]
        test_normal  = normals[test_idx]
        trial_normal = normals[trial_idx]
        tv1  = edge_shared_vertices[pair_idx, 0]
        tv2  = edge_shared_vertices[pair_idx, 1]
        trv1 = edge_shared_vertices[pair_idx, 2]
        trv2 = edge_shared_vertices[pair_idx, 3]

        result = compute_edge_adjacent_adjoint_double_layer_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, tv2, trv1, trv2, order=singular_order, space=space)

        test_points    = phys_points[test_idx]
        test_normals_q = normals_at_quad[test_idx]
        test_int_elem  = integration_elements[test_idx]
        trial_int_elem = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[trial_idx], reflection)
            result = result + _compute_adjoint_double_layer_contribution(
                test_points, reflected_trial_points, test_normals_q,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    edge_matrices = vmap(compute_edge_singular)(jnp.arange(n_edge_pairs))

    # =========================================================================
    # STEP 4: Vertex-adjacent singular contributions
    # =========================================================================
    def compute_vertex_singular(pair_idx):
        test_idx  = vertex_test_indices[pair_idx]
        trial_idx = vertex_trial_indices[pair_idx]
        test_verts   = vertices[faces[test_idx]]
        trial_verts  = vertices[faces[trial_idx]]
        test_normal  = normals[test_idx]
        trial_normal = normals[trial_idx]
        tv1  = vertex_shared_vertices[pair_idx, 0]
        trv1 = vertex_shared_vertices[pair_idx, 1]

        result = compute_vertex_adjacent_adjoint_double_layer_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, trv1, order=singular_order, space=space)

        test_points    = phys_points[test_idx]
        test_normals_q = normals_at_quad[test_idx]
        test_int_elem  = integration_elements[test_idx]
        trial_int_elem = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[trial_idx], reflection)
            result = result + _compute_adjoint_double_layer_contribution(
                test_points, reflected_trial_points, test_normals_q,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    vertex_matrices = vmap(compute_vertex_singular)(jnp.arange(n_vertex_pairs))

    # =========================================================================
    # STEP 5: Scatter
    # =========================================================================
    operator = jnp.zeros((n_dofs, n_dofs), dtype=COMPLEX_DTYPE)

    reg_t = jnp.broadcast_to(local2global[regular_test_indices][:,  :, None], (n_regular_pairs, n_local, n_local)).ravel()
    reg_r = jnp.broadcast_to(local2global[regular_trial_indices][:, None, :], (n_regular_pairs, n_local, n_local)).ravel()
    operator = operator.at[reg_t, reg_r].add(regular_matrices.ravel())

    diag_t = jnp.broadcast_to(local2global[:, :, None], (n_faces, n_local, n_local)).ravel()
    diag_r = jnp.broadcast_to(local2global[:, None, :], (n_faces, n_local, n_local)).ravel()
    operator = operator.at[diag_t, diag_r].add(coincident_matrices.ravel())

    edge_t = jnp.broadcast_to(local2global[edge_test_indices][:,  :, None], (n_edge_pairs, n_local, n_local)).ravel()
    edge_r = jnp.broadcast_to(local2global[edge_trial_indices][:, None, :], (n_edge_pairs, n_local, n_local)).ravel()
    operator = operator.at[edge_t, edge_r].add(edge_matrices.ravel())

    vert_t = jnp.broadcast_to(local2global[vertex_test_indices][:,  :, None], (n_vertex_pairs, n_local, n_local)).ravel()
    vert_r = jnp.broadcast_to(local2global[vertex_trial_indices][:, None, :], (n_vertex_pairs, n_local, n_local)).ravel()
    operator = operator.at[vert_t, vert_r].add(vertex_matrices.ravel())

    return operator


#%% Burton-Miller LHS Assembly (merged K - 0.5*M + eta*W)

def assemble_bm(vertices, faces, normals, k0, eta, adjacency_data,
                quad_order=4, singular_order=4, symmetry=None, space='P1'):
    """
    Assemble Burton-Miller LHS matrix: lhs = K - 0.5*M + eta*W

    Single-pass assembly: geometry is computed once and shared across all
    operators, so peak memory is n_dofs² rather than 3×n_dofs².

    Args:
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        normals:        [F, 3] element normals
        k0:             wavenumber
        eta:            Burton-Miller coupling parameter (complex scalar)
        adjacency_data: pre-computed adjacency lists from load_mesh()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        space:          'P1' (default) or 'DP0'

    Returns:
        lhs: [n_dofs, n_dofs] complex Burton-Miller system matrix
    """
    (regular_test_indices, regular_trial_indices,
     edge_test_indices, edge_trial_indices,
     vertex_test_indices, vertex_trial_indices,
     edge_shared_vertices, vertex_shared_vertices) = adjacency_data

    symmetry_tuple = (False, False, False) if symmetry is None else tuple(bool(s) for s in symmetry)

    return _assemble_bm_jit(
        vertices, faces, normals, k0, eta,
        regular_test_indices, regular_trial_indices,
        edge_test_indices, edge_trial_indices,
        vertex_test_indices, vertex_trial_indices,
        edge_shared_vertices, vertex_shared_vertices,
        quad_order, singular_order, symmetry_tuple, space
    )


@partial(jit, static_argnames=['quad_order', 'singular_order', 'symmetry', 'space'])
def _assemble_bm_jit(vertices, faces, normals, k0, eta,
                      regular_test_indices, regular_trial_indices,
                      edge_test_indices, edge_trial_indices,
                      vertex_test_indices, vertex_trial_indices,
                      edge_shared_vertices, vertex_shared_vertices,
                      quad_order, singular_order, symmetry, space):
    """JIT-compiled Burton-Miller assembly: lhs = K - 0.5*M + eta*W"""
    n_faces = faces.shape[0]
    n_local = 3 if space == 'P1' else 1
    n_dofs  = vertices.shape[0] if space == 'P1' else n_faces
    local2global = faces if space == 'P1' else jnp.arange(n_faces, dtype=jnp.int32)[:, None]

    n_regular_pairs = regular_test_indices.shape[0]
    n_edge_pairs    = edge_test_indices.shape[0]
    n_vertex_pairs  = vertex_test_indices.shape[0]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    basis_vals = p1_basis_functions(quad_points) if space == 'P1' else dp0_basis_functions(quad_points)

    # Geometry — computed once, shared by all three operators
    jacobians = compute_jacobians(vertices, faces)
    integration_elements = compute_integration_elements(jacobians)
    # Surface curls: [F, 3, 3] for P1; [F, 1, 3] zeros for DP0 (constant basis → zero curl)
    surface_curls = (compute_surface_curls(jacobians, normals)
                     if space == 'P1' else jnp.zeros((n_faces, 1, 3)))
    phys_points = compute_element_quadrature_points(vertices, faces, jacobians, quad_points)
    normals_at_quad = jnp.tile(normals[:, None, :], (1, len(quad_weights), 1))  # [F, Q, 3]

    active_reflections = get_active_reflections(symmetry)

    # =========================================================================
    # STEP 1: Regular pairs — dl + eta*hs (no singularity, no mass contribution)
    # =========================================================================
    def compute_regular_pair(pair_idx):
        test_idx = regular_test_indices[pair_idx]
        trial_idx = regular_trial_indices[pair_idx]

        test_points    = phys_points[test_idx]
        test_int_elem  = integration_elements[test_idx]
        test_normal    = normals[test_idx]
        test_curl      = surface_curls[test_idx]
        trial_points   = phys_points[trial_idx]
        trial_int_elem = integration_elements[trial_idx]
        trial_normals_q = normals_at_quad[trial_idx]
        trial_normal   = normals[trial_idx]
        trial_curl     = surface_curls[trial_idx]

        dl_test  = {'points': test_points, 'int_elem': test_int_elem}
        dl_trial = {'points': trial_points, 'normals': trial_normals_q, 'int_elem': trial_int_elem}
        hs_test  = {'points': test_points, 'normals': test_normal, 'curl': test_curl, 'int_elem': test_int_elem}
        hs_trial = {'points': trial_points, 'normals': trial_normal, 'curl': trial_curl, 'int_elem': trial_int_elem}

        dl_local = compute_double_layer_local_matrix(dl_test, dl_trial, k0, basis_vals, quad_weights, symmetry)
        hs_local = compute_hypersingular_local_matrix(hs_test, hs_trial, k0, basis_vals, quad_weights, symmetry)
        return dl_local + eta * hs_local

    regular_matrices = vmap(compute_regular_pair)(jnp.arange(n_regular_pairs))  # [n_regular, 3, 3]

    # =========================================================================
    # STEP 2: Coincident pairs — dl + eta*hs (singular) - 0.5*mass
    # Mass is element-local, so all mass contributions live here.
    # =========================================================================
    def compute_coincident(elem_idx):
        elem_verts  = vertices[faces[elem_idx]]
        elem_normal = normals[elem_idx]
        int_elem    = integration_elements[elem_idx]
        test_curl   = surface_curls[elem_idx]
        trial_curl  = surface_curls[elem_idx]

        dl_result = compute_coincident_double_layer_matrix(
            elem_verts, elem_normal, k0, order=singular_order, space=space)
        hs_result = compute_coincident_hypersingular_matrix(
            elem_verts, elem_normal, k0, order=singular_order, space=space)

        # Local mass: M_ij = Σ_q w_q φ_i(q) φ_j(q) |J|
        weighted_basis = basis_vals * (quad_weights * int_elem)
        mass_local = basis_vals @ weighted_basis.T  # [n_local, n_local]

        test_points    = phys_points[elem_idx]
        test_int_elem  = integration_elements[elem_idx]
        trial_normals_q = normals_at_quad[elem_idx]
        trial_int_elem = integration_elements[elem_idx]
        trial_normal   = normals[elem_idx]

        for reflection in active_reflections:
            reflected_trial_points  = reflect_points(phys_points[elem_idx], reflection)
            reflected_trial_normals = reflect_normals(trial_normals_q, reflection)
            reflected_trial_normal  = reflect_normals(trial_normal, reflection)
            reflected_trial_curl    = reflect_curl(trial_curl, reflection)
            dl_result = dl_result + _compute_double_layer_contribution(
                test_points, reflected_trial_points, reflected_trial_normals,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
            hs_result = hs_result + _compute_hypersingular_contribution(
                test_points, reflected_trial_points, elem_normal, reflected_trial_normal,
                test_curl, reflected_trial_curl, test_int_elem, trial_int_elem, k0,
                basis_vals, quad_weights)

        return dl_result + eta * hs_result - 0.5 * mass_local

    coincident_matrices = vmap(compute_coincident)(jnp.arange(n_faces))

    # =========================================================================
    # STEP 3: Edge-adjacent pairs — dl + eta*hs (singular, no mass)
    # =========================================================================
    def compute_edge_pair(pair_idx):
        test_idx  = edge_test_indices[pair_idx]
        trial_idx = edge_trial_indices[pair_idx]
        test_verts   = vertices[faces[test_idx]]
        trial_verts  = vertices[faces[trial_idx]]
        test_normal  = normals[test_idx]
        trial_normal = normals[trial_idx]
        test_curl    = surface_curls[test_idx]
        trial_curl   = surface_curls[trial_idx]
        tv1  = edge_shared_vertices[pair_idx, 0]
        tv2  = edge_shared_vertices[pair_idx, 1]
        trv1 = edge_shared_vertices[pair_idx, 2]
        trv2 = edge_shared_vertices[pair_idx, 3]

        dl_result = compute_edge_adjacent_double_layer_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, tv2, trv1, trv2, order=singular_order, space=space)
        hs_result = compute_edge_adjacent_hypersingular_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, tv2, trv1, trv2, order=singular_order, space=space)

        test_points     = phys_points[test_idx]
        test_int_elem   = integration_elements[test_idx]
        trial_normals_q = normals_at_quad[trial_idx]
        trial_int_elem  = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points  = reflect_points(phys_points[trial_idx], reflection)
            reflected_trial_normals = reflect_normals(trial_normals_q, reflection)
            reflected_trial_normal  = reflect_normals(trial_normal, reflection)
            reflected_trial_curl    = reflect_curl(trial_curl, reflection)
            dl_result = dl_result + _compute_double_layer_contribution(
                test_points, reflected_trial_points, reflected_trial_normals,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
            hs_result = hs_result + _compute_hypersingular_contribution(
                test_points, reflected_trial_points, test_normal, reflected_trial_normal,
                test_curl, reflected_trial_curl, test_int_elem, trial_int_elem, k0,
                basis_vals, quad_weights)

        return dl_result + eta * hs_result

    edge_matrices = vmap(compute_edge_pair)(jnp.arange(n_edge_pairs))

    # =========================================================================
    # STEP 4: Vertex-adjacent pairs — dl + eta*hs (singular, no mass)
    # =========================================================================
    def compute_vertex_pair(pair_idx):
        test_idx  = vertex_test_indices[pair_idx]
        trial_idx = vertex_trial_indices[pair_idx]
        test_verts   = vertices[faces[test_idx]]
        trial_verts  = vertices[faces[trial_idx]]
        test_normal  = normals[test_idx]
        trial_normal = normals[trial_idx]
        test_curl    = surface_curls[test_idx]
        trial_curl   = surface_curls[trial_idx]
        tv1  = vertex_shared_vertices[pair_idx, 0]
        trv1 = vertex_shared_vertices[pair_idx, 1]

        dl_result = compute_vertex_adjacent_double_layer_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, trv1, order=singular_order, space=space)
        hs_result = compute_vertex_adjacent_hypersingular_matrix(
            test_verts, trial_verts, test_normal, trial_normal, k0,
            tv1, trv1, order=singular_order, space=space)

        test_points     = phys_points[test_idx]
        test_int_elem   = integration_elements[test_idx]
        trial_normals_q = normals_at_quad[trial_idx]
        trial_int_elem  = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points  = reflect_points(phys_points[trial_idx], reflection)
            reflected_trial_normals = reflect_normals(trial_normals_q, reflection)
            reflected_trial_normal  = reflect_normals(trial_normal, reflection)
            reflected_trial_curl    = reflect_curl(trial_curl, reflection)
            dl_result = dl_result + _compute_double_layer_contribution(
                test_points, reflected_trial_points, reflected_trial_normals,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
            hs_result = hs_result + _compute_hypersingular_contribution(
                test_points, reflected_trial_points, test_normal, reflected_trial_normal,
                test_curl, reflected_trial_curl, test_int_elem, trial_int_elem, k0,
                basis_vals, quad_weights)

        return dl_result + eta * hs_result

    vertex_matrices = vmap(compute_vertex_pair)(jnp.arange(n_vertex_pairs))

    # =========================================================================
    # STEP 5: Scatter all contributions into lhs
    # =========================================================================
    lhs = jnp.zeros((n_dofs, n_dofs), dtype=COMPLEX_DTYPE)

    reg_t = jnp.broadcast_to(local2global[regular_test_indices][:,  :, None], (n_regular_pairs, n_local, n_local)).ravel()
    reg_r = jnp.broadcast_to(local2global[regular_trial_indices][:, None, :], (n_regular_pairs, n_local, n_local)).ravel()
    lhs = lhs.at[reg_t, reg_r].add(regular_matrices.ravel())

    diag_t = jnp.broadcast_to(local2global[:, :, None], (n_faces, n_local, n_local)).ravel()
    diag_r = jnp.broadcast_to(local2global[:, None, :], (n_faces, n_local, n_local)).ravel()
    lhs = lhs.at[diag_t, diag_r].add(coincident_matrices.ravel())

    edge_t = jnp.broadcast_to(local2global[edge_test_indices][:,  :, None], (n_edge_pairs, n_local, n_local)).ravel()
    edge_r = jnp.broadcast_to(local2global[edge_trial_indices][:, None, :], (n_edge_pairs, n_local, n_local)).ravel()
    lhs = lhs.at[edge_t, edge_r].add(edge_matrices.ravel())

    vert_t = jnp.broadcast_to(local2global[vertex_test_indices][:,  :, None], (n_vertex_pairs, n_local, n_local)).ravel()
    vert_r = jnp.broadcast_to(local2global[vertex_trial_indices][:, None, :], (n_vertex_pairs, n_local, n_local)).ravel()
    lhs = lhs.at[vert_t, vert_r].add(vertex_matrices.ravel())

    return lhs


#%% Single Layer Assembly

@jit
def _compute_single_layer_contribution(test_points, trial_points,
                                        test_int_elem, trial_int_elem, k0,
                                        basis_vals, quad_weights):
    """Compute single layer contribution for one element pair (no symmetry)."""
    kernel_matrix = helmholtz_single_layer_kernel(test_points, trial_points, k0)
    weights_2d = (quad_weights[:, None] * quad_weights[None, :] *
                  test_int_elem * trial_int_elem)
    weighted_kernel = kernel_matrix * weights_2d
    return jnp.einsum('ip,jq,pq->ij', basis_vals, basis_vals, weighted_kernel)


def compute_single_layer_local_matrix(test_elem_data, trial_elem_data, k0,
                                       basis_vals, quad_weights, symmetry):
    """
    Compute local 3x3 single layer matrix for one element pair.

    For symmetry planes (rigid wall), image sources have the same sign as the
    direct source, so reflected trial contributions are added directly.
    """
    test_points    = test_elem_data['points']
    trial_points   = trial_elem_data['points']
    test_int_elem  = test_elem_data['int_elem']
    trial_int_elem = trial_elem_data['int_elem']

    local_matrix = _compute_single_layer_contribution(
        test_points, trial_points, test_int_elem, trial_int_elem,
        k0, basis_vals, quad_weights)

    for reflection in get_active_reflections(symmetry):
        reflected_trial_points = reflect_points(trial_points, reflection)
        local_matrix = local_matrix + _compute_single_layer_contribution(
            test_points, reflected_trial_points, test_int_elem, trial_int_elem,
            k0, basis_vals, quad_weights)

    return local_matrix


def assemble_single_layer(vertices, faces, k0, adjacency_data,
                          quad_order=4, singular_order=4, symmetry=None, space='P1'):
    """
    Assemble single layer operator V.

    V[i,j] = ∫∫ G(x,y) φ_i(x) φ_j(y) dS_x dS_y,  G = exp(ik|x-y|)/(4π|x-y|)

    Args:
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        k0:             wavenumber
        adjacency_data: pre-computed adjacency lists from compute_adjacency_lists()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        space:          'P1' (default) or 'DP0'
    """
    (regular_test_indices, regular_trial_indices,
     edge_test_indices, edge_trial_indices,
     vertex_test_indices, vertex_trial_indices,
     edge_shared_vertices, vertex_shared_vertices) = adjacency_data

    symmetry_tuple = (False, False, False) if symmetry is None else tuple(bool(s) for s in symmetry)

    return _assemble_single_layer_jit(
        vertices, faces, k0,
        regular_test_indices, regular_trial_indices,
        edge_test_indices, edge_trial_indices,
        vertex_test_indices, vertex_trial_indices,
        edge_shared_vertices, vertex_shared_vertices,
        quad_order, singular_order, symmetry_tuple, space
    )


@partial(jit, static_argnames=['quad_order', 'singular_order', 'symmetry', 'space'])
def _assemble_single_layer_jit(vertices, faces, k0,
                                regular_test_indices, regular_trial_indices,
                                edge_test_indices, edge_trial_indices,
                                vertex_test_indices, vertex_trial_indices,
                                edge_shared_vertices, vertex_shared_vertices,
                                quad_order, singular_order, symmetry, space):
    """JIT-compiled single layer assembly with pre-computed adjacency lists."""
    n_faces = faces.shape[0]
    n_local = 3 if space == 'P1' else 1
    n_dofs  = vertices.shape[0] if space == 'P1' else n_faces
    local2global = faces if space == 'P1' else jnp.arange(n_faces, dtype=jnp.int32)[:, None]

    n_regular_pairs = regular_test_indices.shape[0]
    n_edge_pairs    = edge_test_indices.shape[0]
    n_vertex_pairs  = vertex_test_indices.shape[0]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    basis_vals = p1_basis_functions(quad_points) if space == 'P1' else dp0_basis_functions(quad_points)

    jacobians            = compute_jacobians(vertices, faces)
    integration_elements = compute_integration_elements(jacobians)
    phys_points          = compute_element_quadrature_points(vertices, faces, jacobians, quad_points)

    active_reflections = get_active_reflections(symmetry)

    # =========================================================================
    # STEP 1: Regular (far-field) contributions for non-adjacent pairs
    # =========================================================================
    def compute_regular_pair(pair_idx):
        test_idx  = regular_test_indices[pair_idx]
        trial_idx = regular_trial_indices[pair_idx]

        test_data  = {'points': phys_points[test_idx],  'int_elem': integration_elements[test_idx]}
        trial_data = {'points': phys_points[trial_idx], 'int_elem': integration_elements[trial_idx]}

        return compute_single_layer_local_matrix(test_data, trial_data, k0, basis_vals, quad_weights, symmetry)

    regular_matrices = vmap(compute_regular_pair)(jnp.arange(n_regular_pairs))

    # =========================================================================
    # STEP 2: Coincident singular contributions
    # =========================================================================
    def compute_coincident(elem_idx):
        elem_verts = vertices[faces[elem_idx]]
        result = compute_coincident_single_layer_matrix(elem_verts, k0, order=singular_order, space=space)

        test_points    = phys_points[elem_idx]
        test_int_elem  = integration_elements[elem_idx]
        trial_int_elem = integration_elements[elem_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[elem_idx], reflection)
            result = result + _compute_single_layer_contribution(
                test_points, reflected_trial_points,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    coincident_matrices = vmap(compute_coincident)(jnp.arange(n_faces))

    # =========================================================================
    # STEP 3: Edge-adjacent singular contributions
    # =========================================================================
    def compute_edge_singular(pair_idx):
        test_idx  = edge_test_indices[pair_idx]
        trial_idx = edge_trial_indices[pair_idx]
        test_verts  = vertices[faces[test_idx]]
        trial_verts = vertices[faces[trial_idx]]
        tv1  = edge_shared_vertices[pair_idx, 0]
        tv2  = edge_shared_vertices[pair_idx, 1]
        trv1 = edge_shared_vertices[pair_idx, 2]
        trv2 = edge_shared_vertices[pair_idx, 3]

        result = compute_edge_adjacent_single_layer_matrix(
            test_verts, trial_verts, k0, tv1, tv2, trv1, trv2, order=singular_order, space=space)

        test_points    = phys_points[test_idx]
        test_int_elem  = integration_elements[test_idx]
        trial_int_elem = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[trial_idx], reflection)
            result = result + _compute_single_layer_contribution(
                test_points, reflected_trial_points,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    edge_matrices = vmap(compute_edge_singular)(jnp.arange(n_edge_pairs))

    # =========================================================================
    # STEP 4: Vertex-adjacent singular contributions
    # =========================================================================
    def compute_vertex_singular(pair_idx):
        test_idx  = vertex_test_indices[pair_idx]
        trial_idx = vertex_trial_indices[pair_idx]
        test_verts  = vertices[faces[test_idx]]
        trial_verts = vertices[faces[trial_idx]]
        tv1  = vertex_shared_vertices[pair_idx, 0]
        trv1 = vertex_shared_vertices[pair_idx, 1]

        result = compute_vertex_adjacent_single_layer_matrix(
            test_verts, trial_verts, k0, tv1, trv1, order=singular_order, space=space)

        test_points    = phys_points[test_idx]
        test_int_elem  = integration_elements[test_idx]
        trial_int_elem = integration_elements[trial_idx]

        for reflection in active_reflections:
            reflected_trial_points = reflect_points(phys_points[trial_idx], reflection)
            result = result + _compute_single_layer_contribution(
                test_points, reflected_trial_points,
                test_int_elem, trial_int_elem, k0, basis_vals, quad_weights)
        return result

    vertex_matrices = vmap(compute_vertex_singular)(jnp.arange(n_vertex_pairs))

    # =========================================================================
    # STEP 5: Scatter
    # =========================================================================
    operator = jnp.zeros((n_dofs, n_dofs), dtype=COMPLEX_DTYPE)

    reg_t = jnp.broadcast_to(local2global[regular_test_indices][:,  :, None], (n_regular_pairs, n_local, n_local)).ravel()
    reg_r = jnp.broadcast_to(local2global[regular_trial_indices][:, None, :], (n_regular_pairs, n_local, n_local)).ravel()
    operator = operator.at[reg_t, reg_r].add(regular_matrices.ravel())

    diag_t = jnp.broadcast_to(local2global[:, :, None], (n_faces, n_local, n_local)).ravel()
    diag_r = jnp.broadcast_to(local2global[:, None, :], (n_faces, n_local, n_local)).ravel()
    operator = operator.at[diag_t, diag_r].add(coincident_matrices.ravel())

    edge_t = jnp.broadcast_to(local2global[edge_test_indices][:,  :, None], (n_edge_pairs, n_local, n_local)).ravel()
    edge_r = jnp.broadcast_to(local2global[edge_trial_indices][:, None, :], (n_edge_pairs, n_local, n_local)).ravel()
    operator = operator.at[edge_t, edge_r].add(edge_matrices.ravel())

    vert_t = jnp.broadcast_to(local2global[vertex_test_indices][:,  :, None], (n_vertex_pairs, n_local, n_local)).ravel()
    vert_r = jnp.broadcast_to(local2global[vertex_trial_indices][:, None, :], (n_vertex_pairs, n_local, n_local)).ravel()
    operator = operator.at[vert_t, vert_r].add(vertex_matrices.ravel())

    return operator