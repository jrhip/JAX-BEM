"""
Boundary integral operator assembly.

Every operator is assembled in two passes, following bempp's dense assembler:

  1. a block of test elements at a time, integrated against every trial element
     with the regular quadrature rule and accumulated straight into the
     operator matrix;
  2. a correction on the element pairs that touch, where the regular rule is
     invalid and the Duffy-transformed singular rules take over.

No array is ever indexed by element pair, so peak memory is the operator
matrix plus one block — and the block size comes from a memory budget rather
than from the mesh size.  Both passes are wrapped in jax.checkpoint so reverse
mode recomputes blocks instead of storing them.
"""

import jax.numpy as jnp
from jax import jit, lax, checkpoint
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
                            p1_basis_functions,
                            dp0_basis_functions,
                            get_triangle_quadrature,
                            )
from core.JAX_BEM_kernels import (element_geometry,
                                  reflect_geometry,
                                  select_elements,
                                  regular_block,
                                  choose_block_size,
                                  choose_pair_batch_size,
                                  # re-exported: callers outside this module use these
                                  reflect_points,
                                  reflect_normals,
                                  reflect_curl,
                                  get_active_reflections,
                                  )

from core.JAX_BEM_config import COMPLEX_DTYPE, FLOAT_DTYPE


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

    quad_points, quad_weights = get_triangle_quadrature(quad_order)

    if space == 'P1':
        local2global = faces                                            # [F, 3]
        basis_values = p1_basis_functions(quad_points)
    else:
        local2global = jnp.arange(n_faces, dtype=jnp.int32)[:, None]    # [F, 1]
        basis_values = dp0_basis_functions(quad_points)

    integration_elements = compute_integration_elements(compute_jacobians(vertices, faces))
    weights = quad_weights[None, :] * integration_elements[:, None]     # [F, Q]

    local_mass = jnp.einsum('ip,jp,fp->fij', basis_values, basis_values, weights)

    mass = jnp.zeros((n_dofs, n_dofs), dtype=FLOAT_DTYPE)
    return mass.at[local2global[:, :, None], local2global[:, None, :]].add(local_mass)


#%% Near-Field Singular Quadrature

def _coincident_singular(operator, element_vertices, element_normal, k0, eta, order, space):
    """Exact local matrix for an element integrated against itself."""
    if operator == 'single':
        return compute_coincident_single_layer_matrix(
            element_vertices, k0, order=order, space=space)
    if operator == 'double':
        return compute_coincident_double_layer_matrix(
            element_vertices, element_normal, k0, order=order, space=space)
    if operator == 'adjoint':
        return compute_coincident_adjoint_double_layer_matrix(
            element_vertices, element_normal, k0, order=order, space=space)
    if operator == 'hyper':
        return compute_coincident_hypersingular_matrix(
            element_vertices, element_normal, k0, order=order, space=space)
    if operator == 'bm':
        return (compute_coincident_double_layer_matrix(
                    element_vertices, element_normal, k0, order=order, space=space)
                + eta * compute_coincident_hypersingular_matrix(
                    element_vertices, element_normal, k0, order=order, space=space))
    raise ValueError(f"Unknown operator {operator!r}.")


def _edge_adjacent_singular(operator, test_vertices, trial_vertices,
                            test_normal, trial_normal, k0, eta, shared, order, space):
    """Exact local matrix for two elements sharing an edge."""
    test_v1, test_v2, trial_v1, trial_v2 = shared[0], shared[1], shared[2], shared[3]
    if operator == 'single':
        return compute_edge_adjacent_single_layer_matrix(
            test_vertices, trial_vertices, k0,
            test_v1, test_v2, trial_v1, trial_v2, order=order, space=space)
    if operator == 'double':
        return compute_edge_adjacent_double_layer_matrix(
            test_vertices, trial_vertices, test_normal, trial_normal, k0,
            test_v1, test_v2, trial_v1, trial_v2, order=order, space=space)
    if operator == 'adjoint':
        return compute_edge_adjacent_adjoint_double_layer_matrix(
            test_vertices, trial_vertices, test_normal, trial_normal, k0,
            test_v1, test_v2, trial_v1, trial_v2, order=order, space=space)
    if operator == 'hyper':
        return compute_edge_adjacent_hypersingular_matrix(
            test_vertices, trial_vertices, test_normal, trial_normal, k0,
            test_v1, test_v2, trial_v1, trial_v2, order=order, space=space)
    if operator == 'bm':
        return (compute_edge_adjacent_double_layer_matrix(
                    test_vertices, trial_vertices, test_normal, trial_normal, k0,
                    test_v1, test_v2, trial_v1, trial_v2, order=order, space=space)
                + eta * compute_edge_adjacent_hypersingular_matrix(
                    test_vertices, trial_vertices, test_normal, trial_normal, k0,
                    test_v1, test_v2, trial_v1, trial_v2, order=order, space=space))
    raise ValueError(f"Unknown operator {operator!r}.")


def _vertex_adjacent_singular(operator, test_vertices, trial_vertices,
                              test_normal, trial_normal, k0, eta, shared, order, space):
    """Exact local matrix for two elements sharing a single vertex."""
    test_v, trial_v = shared[0], shared[1]
    if operator == 'single':
        return compute_vertex_adjacent_single_layer_matrix(
            test_vertices, trial_vertices, k0, test_v, trial_v, order=order, space=space)
    if operator == 'double':
        return compute_vertex_adjacent_double_layer_matrix(
            test_vertices, trial_vertices, test_normal, trial_normal, k0,
            test_v, trial_v, order=order, space=space)
    if operator == 'adjoint':
        return compute_vertex_adjacent_adjoint_double_layer_matrix(
            test_vertices, trial_vertices, test_normal, trial_normal, k0,
            test_v, trial_v, order=order, space=space)
    if operator == 'hyper':
        return compute_vertex_adjacent_hypersingular_matrix(
            test_vertices, trial_vertices, test_normal, trial_normal, k0,
            test_v, trial_v, order=order, space=space)
    if operator == 'bm':
        return (compute_vertex_adjacent_double_layer_matrix(
                    test_vertices, trial_vertices, test_normal, trial_normal, k0,
                    test_v, trial_v, order=order, space=space)
                + eta * compute_vertex_adjacent_hypersingular_matrix(
                    test_vertices, trial_vertices, test_normal, trial_normal, k0,
                    test_v, trial_v, order=order, space=space))
    raise ValueError(f"Unknown operator {operator!r}.")


def _near_correction(exact_local_matrix, operator, geometry, test_indices, trial_indices,
                     k0, eta, basis_values, n_singular_points):
    """
    Difference between the exact singular quadrature and the regular quadrature
    that the block pass already applied to these pairs.

    The block pass integrates *every* pair with the regular rule, which is
    wrong wherever two elements touch.  Subtracting the same regular value that
    was added and adding the Duffy value puts those entries right; because both
    terms run through the same kernel code, the erroneous part cancels exactly.

    Image contributions from symmetry planes are not corrected: the mirrored
    element is a different element in space, so the regular rule applies to it.

    Returns:
        values: [n_pairs, L, L] correction for each pair
    """
    n_pairs = test_indices.shape[0]

    @checkpoint  # recompute in the backward pass rather than storing every rule
    def correction(pair_index):
        test_index = test_indices[pair_index]
        trial_index = trial_indices[pair_index]
        regular = regular_block(operator,
                                select_elements(geometry, test_index[None]),
                                select_elements(geometry, trial_index[None]),
                                k0, eta, basis_values)
        exact = exact_local_matrix(pair_index, test_index, trial_index)
        return (exact - regular[0, :, 0, :]).astype(COMPLEX_DTYPE)

    batch_size = choose_pair_batch_size(n_pairs, n_singular_points)
    return lax.map(correction, jnp.arange(n_pairs, dtype=jnp.int32), batch_size=batch_size)


#%% Assembly Driver

@partial(jit, static_argnames=['operator', 'quad_order', 'singular_order',
                               'symmetry', 'space', 'block_size'])
def _assemble(operator, vertices, faces, normals, k0, eta,
              edge_test_indices, edge_trial_indices,
              vertex_test_indices, vertex_trial_indices,
              edge_shared_vertices, vertex_shared_vertices,
              quad_order, singular_order, symmetry, space, block_size):
    """
    Assemble one operator into a dense matrix without forming any per-pair tensor.

    Args:
        operator:       'single', 'double', 'adjoint', 'hyper' or 'bm'
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        normals:        [F, 3] element normals
        k0:             wavenumber
        eta:            Burton-Miller coupling parameter (used by 'bm' only)
        edge_*/vertex_*: near-pair lists from compute_adjacency_lists()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple of 3 bools for (XY, XZ, YZ) planes
        space:          'P1' or 'DP0'
        block_size:     test elements per block, or None to size from the budget

    Returns:
        operator_matrix: [n_dofs, n_dofs] complex weak-form matrix
    """
    n_faces = faces.shape[0]
    n_local = 3 if space == 'P1' else 1
    n_dofs = vertices.shape[0] if space == 'P1' else n_faces
    local2global = faces if space == 'P1' else jnp.arange(n_faces, dtype=jnp.int32)[:, None]

    quad_points, quad_weights = get_triangle_quadrature(quad_order)
    n_quad = quad_weights.shape[0]
    basis_values = (p1_basis_functions(quad_points) if space == 'P1'
                    else dp0_basis_functions(quad_points))

    geometry = element_geometry(vertices, faces, normals, quad_points, quad_weights,
                                space, with_curls=operator in ('hyper', 'bm'))

    # Method of images: each active reflection adds a mirrored copy of the mesh
    # as an extra set of sources, integrated with the same regular rule.
    sources = [geometry] + [reflect_geometry(geometry, reflection)
                            for reflection in get_active_reflections(symmetry)]

    # =========================================================================
    # PASS 1: regular quadrature over every element pair, one block of test
    # elements at a time, accumulated directly into the operator matrix.
    # =========================================================================
    if block_size is None:
        block_size = choose_block_size(n_faces, n_quad)
    n_blocks = -(-n_faces // block_size)

    # Pad the last block by repeating an element and masking it out, so every
    # block has the same shape and lax.scan can compile a single body.
    padded = jnp.arange(n_blocks * block_size, dtype=jnp.int32)
    block_indices = jnp.minimum(padded, n_faces - 1).reshape(n_blocks, block_size)
    block_valid = (padded < n_faces).reshape(n_blocks, block_size)

    @checkpoint  # blocks are recomputed in the backward pass, never stored
    def block_contribution(indices, valid):
        target = select_elements(geometry, indices)
        # Zero the padding at the source: every term is proportional to the
        # test quadrature weights, so a [B, Q] mask does the whole job and no
        # block-sized temporary is needed to apply it.
        target['weights'] = jnp.where(valid[:, None], target['weights'], 0.0)
        return sum(regular_block(operator, target, source, k0, eta, basis_values)
                   for source in sources).astype(COMPLEX_DTYPE)

    def accumulate(operator_matrix, block):
        indices, valid = block
        # Scatter in two stages so both index arrays stay small: the trial map
        # is loop invariant and the test map is one row index per element.
        # A single [B, L, F, L] scatter would instead make reverse mode store
        # broadcast indices of that size for every block.
        columns = jnp.zeros((block_size, n_local, n_dofs), dtype=COMPLEX_DTYPE)
        columns = columns.at[:, :, local2global].add(block_contribution(indices, valid))
        return operator_matrix.at[local2global[indices]].add(columns), None

    operator_matrix, _ = lax.scan(accumulate,
                                  jnp.zeros((n_dofs, n_dofs), dtype=COMPLEX_DTYPE),
                                  (block_indices, block_valid))

    # =========================================================================
    # PASS 2: swap the regular value for the singular one on touching pairs.
    # =========================================================================
    element_vertices = vertices[faces]                            # [F, 3, 3]
    element_indices = jnp.arange(n_faces, dtype=jnp.int32)

    def coincident_exact(pair_index, test_index, trial_index):
        exact = _coincident_singular(operator, element_vertices[test_index],
                                     normals[test_index], k0, eta, singular_order, space)
        if operator == 'bm':
            # The identity term of the Burton-Miller form is element local, so
            # it belongs on the diagonal blocks with the coincident correction.
            local_mass = basis_values @ (basis_values * geometry['weights'][test_index]).T
            exact = exact - 0.5 * local_mass
        return exact

    def edge_exact(pair_index, test_index, trial_index):
        return _edge_adjacent_singular(
            operator, element_vertices[test_index], element_vertices[trial_index],
            normals[test_index], normals[trial_index], k0, eta,
            edge_shared_vertices[pair_index], singular_order, space)

    def vertex_exact(pair_index, test_index, trial_index):
        return _vertex_adjacent_singular(
            operator, element_vertices[test_index], element_vertices[trial_index],
            normals[test_index], normals[trial_index], k0, eta,
            vertex_shared_vertices[pair_index], singular_order, space)

    # Duffy rule sizes, used to batch the correction within the memory budget.
    corrections = [(element_indices, element_indices, coincident_exact,
                    6 * singular_order ** 4)]
    if edge_test_indices.shape[0] > 0:
        corrections.append((edge_test_indices, edge_trial_indices, edge_exact,
                            5 * singular_order ** 4))
    if vertex_test_indices.shape[0] > 0:
        corrections.append((vertex_test_indices, vertex_trial_indices, vertex_exact,
                            2 * singular_order ** 4))

    for test_indices, trial_indices, exact_local_matrix, n_singular_points in corrections:
        values = _near_correction(exact_local_matrix, operator, geometry,
                                  test_indices, trial_indices, k0, eta,
                                  basis_values, n_singular_points)
        rows = local2global[test_indices]                         # [n_pairs, L]
        cols = local2global[trial_indices]                        # [n_pairs, L]
        operator_matrix = operator_matrix.at[rows[:, :, None], cols[:, None, :]].add(values)

    return operator_matrix


def _symmetry_tuple(symmetry):
    """Static, hashable form of the symmetry flags."""
    return (False, False, False) if symmetry is None else tuple(bool(s) for s in symmetry)


#%% Operators

def assemble_single_layer(vertices, faces, k0, adjacency_data,
                          quad_order=4, singular_order=4, symmetry=None,
                          space='P1', block_size=None):
    """
    Assemble single layer operator V.

    V[i,j] = ∫∫ G(x,y) φ_i(x) φ_j(y) dS_x dS_y,  G = exp(ik|x-y|)/(4π|x-y|)

    Args:
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        k0:             wavenumber
        adjacency_data: near-pair lists from compute_adjacency_lists()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        space:          'P1' (default) or 'DP0'
        block_size:     test elements per block, or None to size from the budget
    """
    normals = jnp.zeros((faces.shape[0], 3), dtype=FLOAT_DTYPE)  # unused by V
    return _assemble('single', vertices, faces, normals, k0, 0.0,
                     *adjacency_data,
                     quad_order, singular_order, _symmetry_tuple(symmetry),
                     space, block_size)


def assemble_double_layer(vertices, faces, normals, k0, adjacency_data,
                          quad_order=4, singular_order=4, symmetry=None,
                          space='P1', block_size=None):
    """
    Assemble double layer operator K.

    K[i,j] = ∫∫ ∂G(x,y)/∂n(y) φ_i(x) φ_j(y) dS_x dS_y

    Args:
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        normals:        [F, 3] element normals
        k0:             wavenumber
        adjacency_data: near-pair lists from compute_adjacency_lists()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        space:          'P1' (default) or 'DP0'
        block_size:     test elements per block, or None to size from the budget
    """
    return _assemble('double', vertices, faces, normals, k0, 0.0,
                     *adjacency_data,
                     quad_order, singular_order, _symmetry_tuple(symmetry),
                     space, block_size)


def assemble_adjoint_double_layer(vertices, faces, normals, k0, adjacency_data,
                                  quad_order=4, singular_order=4, symmetry=None,
                                  space='P1', block_size=None):
    """
    Assemble adjoint double layer operator K'.

    K'[i,j] = ∫∫ ∂G(x,y)/∂n(x) φ_i(x) φ_j(y) dS_x dS_y — the normal derivative
    is taken at the test point, so only the trial side is mirrored by symmetry.

    Args: as assemble_double_layer.
    """
    return _assemble('adjoint', vertices, faces, normals, k0, 0.0,
                     *adjacency_data,
                     quad_order, singular_order, _symmetry_tuple(symmetry),
                     space, block_size)


def assemble_hypersingular(vertices, faces, normals, k0, adjacency_data,
                           quad_order=4, singular_order=4, symmetry=None,
                           space='P1', block_size=None):
    """
    Assemble hypersingular operator W.

    Uses Maue's identity, so only the scalar single layer kernel is needed:
    the curl-curl term plus a k² term weighted by the normal product.

    Args: as assemble_double_layer.
    """
    return _assemble('hyper', vertices, faces, normals, k0, 0.0,
                     *adjacency_data,
                     quad_order, singular_order, _symmetry_tuple(symmetry),
                     space, block_size)


def assemble_bm(vertices, faces, normals, k0, eta, adjacency_data,
                quad_order=4, singular_order=4, symmetry=None,
                space='P1', block_size=None):
    """
    Assemble the Burton-Miller LHS matrix: lhs = K - 0.5*M + eta*W

    Single pass over the geometry: the double layer and hypersingular kernels
    share one kernel evaluation per block, and the mass term rides along on the
    coincident correction.

    Args:
        vertices:       [N, 3] vertex positions
        faces:          [F, 3] triangle connectivity
        normals:        [F, 3] element normals
        k0:             wavenumber
        eta:            Burton-Miller coupling parameter (complex scalar)
        adjacency_data: near-pair lists from compute_adjacency_lists()
        quad_order:     quadrature order for regular integration
        singular_order: quadrature order for singular integration
        symmetry:       tuple/array of 3 bools for (XY, XZ, YZ) planes, or None
        space:          'P1' (default) or 'DP0'
        block_size:     test elements per block, or None to size from the budget

    Returns:
        lhs: [n_dofs, n_dofs] complex Burton-Miller system matrix
    """
    return _assemble('bm', vertices, faces, normals, k0, eta,
                     *adjacency_data,
                     quad_order, singular_order, _symmetry_tuple(symmetry),
                     space, block_size)
