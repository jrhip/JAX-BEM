"""
Pairwise Helmholtz kernels, element geometry and block quadrature.

Nothing here is ever sized by the number of element *pairs*.  The assembler
walks blocks of test elements and evaluates one block against every trial
element in a single batched contraction, which is the streaming strategy used
by bempp's dense assembler.  Written as einsums, each block contraction
maps onto a GEMM rather than a batch of tiny 3x3 products.

All four operators are built from one kernel evaluation: the Green's function
and its gradient with respect to the test point, exactly the four scalars
bempp's FMM kernels return.
"""

import jax.numpy as jnp
from jax import jit

from core.JAX_BEM_config import COMPLEX_DTYPE, FLOAT_DTYPE, TILE_BUDGET_MB
from core.JAX_BEM_mesh import (compute_jacobians,
                               compute_integration_elements,
                               compute_surface_curls,
                               compute_element_quadrature_points)

M_INV_4PI = 1.0 / (4.0 * jnp.pi)

# Tile budget in bytes; set TILE_BUDGET_MB in JAX_BEM_config to change it.
TILE_BUDGET_BYTES = TILE_BUDGET_MB << 20


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

    For a Neumann (sound-hard) boundary condition on the symmetry plane, the
    normal component perpendicular to the plane flips sign while tangential
    components stay the same — the same transformation as for points.
    """
    return reflect_points(normals, symmetry)


@jit
def reflect_curl(curl, symmetry):
    """
    Reflect surface curl vectors across active symmetry planes.

    The surface curl is a pseudovector, so it transforms as
    curl' = det(R) * R @ curl, with det(R) = -1 for an odd number of plane
    reflections and +1 for an even number.

    Args:
        curl: [..., 3, 3] surface curl vectors (curl[..., i, :] is basis i)
        symmetry: [3] boolean array for [XY, XZ, YZ] planes
    """
    signs = jnp.array([
        jnp.where(symmetry[2], -1.0, 1.0),  # x sign (YZ plane)
        jnp.where(symmetry[1], -1.0, 1.0),  # y sign (XZ plane)
        jnp.where(symmetry[0], -1.0, 1.0),  # z sign (XY plane)
    ])
    determinant = signs[0] * signs[1] * signs[2]
    return determinant * curl * signs


def get_active_reflections(symmetry_tuple):
    """
    Generate all active reflection combinations for the given symmetry planes.

    Pure Python, called at trace time.  For N active planes this returns
    2^N - 1 combinations (every image source, excluding the original).

    Args:
        symmetry_tuple: tuple of 3 bools for (XY, XZ, YZ) symmetry planes

    Returns:
        List of [3] bool arrays, empty if no plane is active.
    """
    symmetry_xy, symmetry_xz, symmetry_yz = symmetry_tuple
    reflections = []

    if symmetry_xy:
        reflections.append(jnp.array([True, False, False]))
    if symmetry_xz:
        reflections.append(jnp.array([False, True, False]))
    if symmetry_yz:
        reflections.append(jnp.array([False, False, True]))

    if symmetry_xy and symmetry_xz:
        reflections.append(jnp.array([True, True, False]))
    if symmetry_xz and symmetry_yz:
        reflections.append(jnp.array([False, True, True]))
    if symmetry_xy and symmetry_yz:
        reflections.append(jnp.array([True, False, True]))

    if symmetry_xy and symmetry_xz and symmetry_yz:
        reflections.append(jnp.array([True, True, True]))

    return reflections


#%% Element Geometry

def element_geometry(vertices, faces, normals, quad_points, quad_weights,
                     space='P1', with_curls=False):
    """
    Per-element quadrature data, computed once and shared by every operator.

    Args:
        vertices:     [N, 3] vertex positions
        faces:        [F, 3] triangle connectivity
        normals:      [F, 3] element normals
        quad_points:  [2, Q] reference quadrature points
        quad_weights: [Q] quadrature weights
        space:        'P1' or 'DP0'
        with_curls:   compute surface curls (only the hypersingular form needs them)

    Returns:
        dict with
            points  [F, Q, 3] physical quadrature points
            normals [F, 3]    element normals
            curls   [F, L, 3] surface curls (zeros unless with_curls and P1)
            weights [F, Q]    quadrature weights times integration element
    """
    n_faces = faces.shape[0]
    n_local = 3 if space == 'P1' else 1

    jacobians = compute_jacobians(vertices, faces)
    integration_elements = compute_integration_elements(jacobians)

    if with_curls and space == 'P1':
        curls = compute_surface_curls(jacobians, normals)
    else:
        # DP0 basis functions are constant, so their surface curl vanishes.
        curls = jnp.zeros((n_faces, n_local, 3), dtype=FLOAT_DTYPE)

    return {
        'points':  compute_element_quadrature_points(vertices, faces, jacobians, quad_points),
        'normals': normals,
        'curls':   curls,
        'weights': quad_weights[None, :] * integration_elements[:, None],
    }


def reflect_geometry(geometry, reflection):
    """Mirror an element geometry across one combination of symmetry planes."""
    return {
        'points':  reflect_points(geometry['points'], reflection),
        'normals': reflect_normals(geometry['normals'], reflection),
        'curls':   reflect_curl(geometry['curls'], reflection),
        'weights': geometry['weights'],
    }


def select_elements(geometry, indices):
    """Slice a geometry down to a block of elements."""
    return {key: value[indices] for key, value in geometry.items()}


#%% Kernels

@jit
def helmholtz_pair_kernels(target_points, source_points, k0):
    """
    Helmholtz Green's function and radial gradient factor for every point pair.

    Args:
        target_points: [T, 3] evaluation (test) points x
        source_points: [S, 3] source (trial) points y
        k0:            wavenumber

    Returns:
        diff:   [T, S, 3] x - y
        green:  [T, S]    G(x, y) = exp(ik|x-y|) / (4π|x-y|)
        radial: [T, S]    scalar with ∇_x G = diff * radial

    Coincident points (r = 0) return zero rather than infinity, matching
    bempp's kernels; those pairs are replaced by singular quadrature in the
    near-field correction.  The guarded reciprocal also keeps the derivative
    with respect to vertex positions finite.
    """
    diff = target_points[:, None, :] - source_points[None, :, :]
    r_squared = jnp.sum(diff * diff, axis=-1)

    coincident = r_squared == 0.0
    r_squared_safe = jnp.where(coincident, 1.0, r_squared)
    r = jnp.sqrt(r_squared_safe)

    green = jnp.where(coincident, 0.0,
                      jnp.exp(1j * k0 * r) * M_INV_4PI / r).astype(COMPLEX_DTYPE)
    radial = green * (1j * k0 * r - 1.0) / r_squared_safe

    return diff, green, radial


#%% Block Quadrature

def regular_block(operator, target, source, k0, eta, basis_values):
    """
    Weak-form local matrices between two sets of elements, regular quadrature.

    This is the whole far-field assembler: one call covers a block of test
    elements against every trial element, so no per-pair tensor is ever built.
    Singular pairs are integrated here too — with the wrong rule — and put
    right afterwards by the near-field correction, exactly as bempp's FMM
    assembler corrects the near part of its kernel sum.

    Args:
        operator:     one of 'single', 'double', 'adjoint', 'hyper', 'bm'
        target:       geometry dict for T test elements (see element_geometry)
        source:       geometry dict for S trial elements
        k0:           wavenumber
        eta:          Burton-Miller coupling parameter (only used by 'bm')
        basis_values: [L, Q] basis functions at the quadrature points

    Returns:
        block: [T, L, S, L] local matrices for every (test, trial) pair
    """
    n_test, n_quad = target['weights'].shape
    n_trial = source['weights'].shape[0]

    diff, green, radial = helmholtz_pair_kernels(
        target['points'].reshape(-1, 3), source['points'].reshape(-1, 3), k0)

    block_shape = (n_test, n_quad, n_trial, n_quad)
    diff = diff.reshape(*block_shape, 3)
    green = green.reshape(block_shape)
    radial = radial.reshape(block_shape)

    # Basis functions premultiplied by their quadrature weight and integration
    # element: the map from element coefficients to quadrature point values.
    test_spread = basis_values[None, :, :] * target['weights'][:, None, :]   # [T, L, Q]
    trial_spread = basis_values[None, :, :] * source['weights'][:, None, :]  # [S, L, Q]

    def weak_form(kernel):
        """[T, Q, S, Q] kernel values -> [T, L, S, L] weak form."""
        return jnp.einsum('tip,sjq,tpsq->tisj', test_spread, trial_spread, kernel)

    def double_layer():
        # ∂G/∂n_y = -∇_x G · n_y
        normal_dot = jnp.einsum('tpsqc,sc->tpsq', diff, source['normals'])
        return weak_form(-normal_dot * radial)

    def adjoint_double_layer():
        # ∂G/∂n_x = ∇_x G · n_x
        normal_dot = jnp.einsum('tpsqc,tc->tpsq', diff, target['normals'])
        return weak_form(normal_dot * radial)

    def hypersingular():
        # Maue's identity: curl-curl term plus a k² mass-like term, both of
        # which need only the scalar single layer kernel.
        kernel_sum = jnp.einsum('tp,sq,tpsq->ts',
                                target['weights'], source['weights'], green)
        curl_term = jnp.einsum('ts,tic,sjc->tisj',
                               kernel_sum, target['curls'], source['curls'])
        normal_product = target['normals'] @ source['normals'].T  # [T, S]
        return curl_term - k0**2 * normal_product[:, None, :, None] * weak_form(green)

    if operator == 'single':
        return weak_form(green)
    if operator == 'double':
        return double_layer()
    if operator == 'adjoint':
        return adjoint_double_layer()
    if operator == 'hyper':
        return hypersingular()
    if operator == 'bm':
        return double_layer() + eta * hypersingular()
    raise ValueError(f"Unknown operator {operator!r}.")


#%% Block Sizing

def choose_block_size(n_faces, n_quad, budget_bytes=None):
    """
    Test elements per assembly tile, sized to the working-memory budget.

    One tile holds a handful of [B*Q, F*Q] arrays (the separation vector, the
    Green's function and its radial factor), so its footprint grows with
    B * F * Q^2 and the block size falls as the mesh grows.
    """
    budget_bytes = TILE_BUDGET_BYTES if budget_bytes is None else budget_bytes
    bytes_per_pair = 8 * jnp.dtype(COMPLEX_DTYPE).itemsize * n_quad * n_quad
    block_size = budget_bytes // (bytes_per_pair * max(n_faces, 1))
    return int(min(n_faces, max(block_size, 1)))


def choose_chunk_size(n_faces, budget_bytes=None):
    """
    Evaluation points per potential-evaluation chunk, from the same budget.

    Field evaluation holds [C, F] arrays for a chunk of C points, so the chunk
    has to shrink as the mesh grows — a fixed chunk size silently turns into
    tens of gigabytes on a fine mesh.
    """
    budget_bytes = TILE_BUDGET_BYTES if budget_bytes is None else budget_bytes
    bytes_per_pair = 8 * jnp.dtype(COMPLEX_DTYPE).itemsize
    return int(max(budget_bytes // (bytes_per_pair * max(n_faces, 1)), 1))


def choose_pair_batch_size(n_pairs, n_quad_points, budget_bytes=None):
    """
    Element pairs per singular-quadrature batch, sized to the same budget.

    Duffy rules use 6*order^4 points for coincident pairs, so batching matters
    here as much as it does for the regular blocks.
    """
    budget_bytes = TILE_BUDGET_BYTES if budget_bytes is None else budget_bytes
    bytes_per_pair = 48 * jnp.dtype(COMPLEX_DTYPE).itemsize * n_quad_points
    batch_size = budget_bytes // bytes_per_pair
    return int(min(max(n_pairs, 1), max(batch_size, 1)))
