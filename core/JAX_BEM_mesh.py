import jax.numpy as jnp
from jax import jit
import numpy as np
from core.JAX_BEM_config import FLOAT_DTYPE

M_INV_4PI = 1.0 / (4.0 * jnp.pi)

"""
Contains mesh processing functions
"""

#%% Symmetry helpers

def cut_mesh_for_symmetry(vertices, elements, normals, symmetry, element_tags=None):
    """
    Cut mesh to keep only elements on positive side of symmetry planes.

    For each active symmetry plane, keeps elements whose centroid is on the
    positive side (>= 0) of the plane. Vertices are renumbered to form a
    compact mesh.

    Args:
        vertices: [N, 3] vertex positions (numpy array)
        elements: [F, 3] triangle connectivity (numpy array)
        normals: [F, 3] element normals (numpy array)
        symmetry: [3] boolean array for [XY, XZ, YZ] planes
        element_tags: [F] optional integer array of element domain indices

    Returns:
        new_vertices: [N', 3] reduced vertex positions
        new_elements: [F', 3] reduced triangle connectivity
        new_normals: [F', 3] reduced element normals
        new_element_tags: [F'] filtered element tags (only if element_tags provided)
    """
    # Compute element centroids
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0  # [F, 3]

    # Start with all elements kept
    keep_mask = np.ones(len(elements), dtype=bool)

    # XY plane (z=0): keep elements with centroid z >= 0
    if symmetry[0]:
        keep_mask &= (centroids[:, 2] >= 0)

    # XZ plane (y=0): keep elements with centroid y >= 0
    if symmetry[1]:
        keep_mask &= (centroids[:, 1] >= 0)

    # YZ plane (x=0): keep elements with centroid x >= 0
    if symmetry[2]:
        keep_mask &= (centroids[:, 0] >= 0)

    # Filter elements, normals, and optional element tags
    new_elements = elements[keep_mask]
    new_normals = normals[keep_mask]
    new_element_tags = element_tags[keep_mask] if element_tags is not None else None

    # Find unique vertices used by remaining elements
    used_vertices = np.unique(new_elements.ravel())

    # Create mapping from old vertex indices to new
    vertex_map = np.full(len(vertices), -1, dtype=np.int32)
    vertex_map[used_vertices] = np.arange(len(used_vertices))

    # Remap element connectivity
    new_elements = vertex_map[new_elements]

    # Extract only used vertices
    new_vertices = vertices[used_vertices]

    if element_tags is not None:
        return new_vertices, new_elements, new_normals, new_element_tags
    return new_vertices, new_elements, new_normals

def create_symmetry_gradient_mask(vertices, symmetry, tol=1e-6):
    """
    Create a mask to zero out gradient components perpendicular to symmetry planes.

    Vertices on a symmetry plane can slide along it, but not move perpendicular.
    Returns a [N, 3] mask where 0 = fixed component, 1 = free component.
    """
    mask = jnp.ones_like(vertices)

    # XZ plane (y=0): fix y-coordinate of vertices with y ≈ 0
    if symmetry[1]:
        on_xz = jnp.abs(vertices[:, 1]) < tol
        mask = mask.at[:, 1].set(jnp.where(on_xz, 0.0, mask[:, 1]))

    # YZ plane (x=0): fix x-coordinate of vertices with x ≈ 0
    if symmetry[2]:
        on_yz = jnp.abs(vertices[:, 0]) < tol
        mask = mask.at[:, 0].set(jnp.where(on_yz, 0.0, mask[:, 0]))

    # XY plane (z=0): fix z-coordinate of vertices with z ≈ 0
    if symmetry[0]:
        on_xy = jnp.abs(vertices[:, 2]) < tol
        mask = mask.at[:, 2].set(jnp.where(on_xy, 0.0, mask[:, 2]))

    return mask

def mirror_mesh(vertices, elements, symmetry, tol=1e-6):
    """
    Mirror mesh across symmetry planes to reconstruct full geometry.

    For each active symmetry plane, mirrors vertices and elements,
    flipping element winding to maintain consistent normals.
    """
    vertices = np.array(vertices)
    elements = np.array(elements)

    for plane_idx, active in enumerate(symmetry):
        if not active:
            continue

        # Determine which coordinate to flip
        # XY plane (idx 0) -> flip z, XZ plane (idx 1) -> flip y, YZ plane (idx 2) -> flip x
        coord_idx = 2 - plane_idx  # Maps: 0->2(z), 1->1(y), 2->0(x)

        # Find vertices NOT on the symmetry plane (to avoid duplicates)
        on_plane = np.abs(vertices[:, coord_idx]) < tol
        not_on_plane = ~on_plane

        # Mirror all vertices
        mirrored_verts = vertices.copy()
        mirrored_verts[:, coord_idx] *= -1

        # New vertex indices for mirrored vertices (offset by current count)
        n_verts = len(vertices)

        # Build mapping: original vertex -> mirrored vertex index
        # Vertices on plane map to themselves, others map to new mirrored vertices
        # Use cumsum to get compact indices for non-on-plane vertices
        new_idx_offset = np.cumsum(not_on_plane) - 1  # 0-indexed offset into new vertices
        new_vert_indices = np.where(on_plane, np.arange(n_verts), n_verts + new_idx_offset)

        # Only add vertices that are not on the plane
        new_verts = mirrored_verts[not_on_plane]

        # Create mirrored elements with flipped winding (swap v1 and v2)
        mirrored_elems = elements.copy()
        mirrored_elems[:, [1, 2]] = mirrored_elems[:, [2, 1]]  # Flip winding
        mirrored_elems = new_vert_indices[mirrored_elems]  # Remap to new vertex indices

        # Combine
        vertices = np.vstack([vertices, new_verts])
        elements = np.vstack([elements, mirrored_elems])

    return vertices, elements

#%% Main mesh loader

def load_mesh(mesh, symmetry):
    """
    Load mesh and pre-compute adjacency data for singular integration.

    Adjacency computation uses NumPy (non-differentiable) but only depends on
    mesh topology which is fixed. Pre-computing here keeps the main solver
    fully differentiable with respect to vertex positions and wavenumber.

    Args:
        mesh: bempp mesh object
        symmetry: [3] boolean array for [XY, XZ, YZ] symmetry planes

    Returns:
        vertices: [N, 3] vertex positions (JAX array)
        elements: [F, 3] triangle connectivity (JAX array)
        normals: [F, 3] element normals (JAX array)
        adjacency_data: tuple of pre-computed adjacency lists for singular integration
        element_tags: [F] integer array of GMSH physical group tags per element (JAX array)
    """
    vertices = mesh.vertices.T
    elements = mesh.elements.T
    normals = mesh.normals
    element_tags = np.asarray(mesh.domain_indices)  # [F] physical group tags

    print(f"  Mesh size: {len(vertices)} vertices, {len(elements)} elements")

    # Convert symmetry to numpy for mesh cutting
    symmetry_np = np.asarray(symmetry)

    # Cut mesh if any symmetry plane is active
    if np.any(symmetry_np):
        vertices, elements, normals, element_tags = cut_mesh_for_symmetry(
            vertices, elements, normals, symmetry_np, element_tags)
        print(f"  Mesh cut for symmetry: {len(vertices)} vertices, {len(elements)} elements")

    # Convert to JAX arrays
    vertices = jnp.array(vertices, dtype=FLOAT_DTYPE)
    elements = jnp.array(elements, dtype=jnp.int32)
    normals = jnp.array(normals, dtype=FLOAT_DTYPE)
    element_tags = jnp.array(element_tags, dtype=jnp.int32)

    # Pre-compute adjacency lists (topology-dependent, non-differentiable)
    # This must happen before entering the differentiable solver
    adjacency_data = compute_adjacency_lists(elements)

    return vertices, elements, normals, adjacency_data, element_tags

#%% Mesh functions

@jit
def compute_normals(vertices, elements):
    """
    Compute unit normals for triangular elements from vertex positions.

    This is differentiable with respect to vertices, allowing gradients
    to flow through normal computation during optimization.

    Args:
        vertices: [N, 3] vertex positions
        elements: [F, 3] triangle indices

    Returns:
        normals: [F, 3] unit normal vectors for each element
    """
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]

    # Edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Cross product gives normal direction
    cross = jnp.cross(edge1, edge2)

    # Normalize to unit length
    norms = jnp.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / (norms + 1e-10)  # small epsilon for numerical stability

    return normals

@jit
def compute_element_centroids(vertices, elements):
    """
    Compute centroids of triangular elements.
    
    Args:
        vertices: [N, 3] vertex positions
        elements: [F, 3] triangle indices
    
    Returns:
        centroids: [F, 3] array of triangle centroids
    """
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]
    
    centroids = (v0 + v1 + v2) / 3.0
    return centroids

@jit
def compute_jacobians(vertices, faces):
    """Compute jacobians [F, 3, 2]."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    return jnp.stack([edge1, edge2], axis=2)

@jit
def compute_integration_elements(jacobians):
    """Compute integration elements (areas)."""
    jac_t_jac = jnp.einsum('fij,fik->fjk', jacobians, jacobians)
    return jnp.sqrt(jnp.linalg.det(jac_t_jac))

@jit
def compute_surface_curls(jacobians, normals):
    """
    Compute surface curls [F, 3, 3].

    Returns surface_curls where surface_curls[f, i, :] is the 3D curl vector
    for basis function i on face f.

    Following bempp convention:
    - reference_gradient is [2, 3]: 2 reference coords × 3 basis functions
    - jac_inv_t is [F, 3, 2]: 3 spatial coords × 2 reference coords
    - surface_gradients = jac_inv_t @ ref_grad gives [F, 3, 3]: spatial × basis
    - surface_curl[i] = cross(normal, surface_gradients[:, i])
    """
    ref_grad = jnp.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])  # [2, 3]

    jac_t_jac = jnp.einsum('fij,fik->fjk', jacobians, jacobians)  # [F, 2, 2]
    jac_t_jac_inv = jnp.linalg.inv(jac_t_jac)  # [F, 2, 2]
    jac_inv_t = jnp.einsum('fij,fjk->fik', jacobians, jac_t_jac_inv)  # [F, 3, 2]

    # surface_gradients[f, spatial, basis] = jac_inv_t @ ref_grad
    surface_gradients = jnp.einsum('fij,jk->fik', jac_inv_t, ref_grad)  # [F, 3, 3]

    # For cross product, we need to cross normal with each column (gradient of each basis fn)
    # Transpose to [F, basis, spatial] so cross product works on last axis (spatial)
    surface_gradients_t = jnp.transpose(surface_gradients, (0, 2, 1))  # [F, 3, 3] = [F, basis, spatial]

    # Now cross: normals[:, None, :] is [F, 1, 3], surface_gradients_t is [F, 3, 3]
    # Cross product on last axis (spatial dimension)
    surface_curls = jnp.cross(normals[:, None, :], surface_gradients_t)  # [F, 3, 3] = [F, basis, spatial]

    return surface_curls

@jit
def p1_basis_functions(local_coords):
    """Evaluate P1 basis functions. Returns [3, Q]."""
    xi = local_coords[0]
    eta = local_coords[1]
    return jnp.array([1.0 - xi - eta, xi, eta])

def dp0_basis_functions(local_coords):
    """Evaluate DP0 basis function. Returns [1, Q] — constant 1 over element."""
    return jnp.ones((1, local_coords.shape[1]))

def get_triangle_quadrature(order=4):
    """Get quadrature points [2, Q] and weights [Q].

    Note: bempp default is order=4 (4 points).
    """
    if order == 1:
        points = jnp.array([[1/3], [1/3]])
        weights = jnp.array([0.5])
    elif order == 3:
        # 3-point rule (degree of precision 2)
        points = jnp.array([
            [1/6, 2/3, 1/6],  # xi
            [1/6, 1/6, 2/3]   # eta
        ])
        weights = jnp.array([1/6, 1/6, 1/6])
    elif order == 4:
        # 4-point rule (degree of precision 3) - bempp default
        # From reference_code/api/integration/triangle_gauss.py
        # Barycentric coords: (1/3,1/3,1/3), (0.6,0.2,0.2), (0.2,0.6,0.2), (0.2,0.2,0.6)
        # (xi, eta) = (bary[1], bary[2])
        points = jnp.array([
            [1/3, 0.2, 0.6, 0.2],  # xi
            [1/3, 0.2, 0.2, 0.6]   # eta
        ])
        # Raw weights: -0.5625, 0.520833..., 0.520833..., 0.520833... (sum=1)
        # Multiply by 0.5 for triangle area
        weights = jnp.array([-0.5625, 0.5208333333333333,
                             0.5208333333333333, 0.5208333333333333]) * 0.5
    elif order == 7:
        a1, a2 = 0.0597158717, 0.7974269853
        b1, b2 = 0.4701420641, 0.1012865073
        w1, w2, w3 = 0.1125, 0.0629695902, 0.0661970763
        points = jnp.array([
            [1/3, a1, a2, a1, b1, b2, b1],  # xi
            [1/3, a1, a1, a2, b1, b1, b2]   # eta
        ])
        weights = jnp.array([w1, w2, w2, w2, w3, w3, w3]) * 0.5
    else:
        raise ValueError(f"Unsupported order: {order}. Use 1, 3, 4, or 7.")
    
    return points, weights

@jit
def compute_element_quadrature_points(vertices, faces, jacobians, quad_points):
    """Compute physical quadrature points [F, Q, 3]."""
    v0 = vertices[faces[:, 0]]
    phys_points = v0[:, None, :] + jnp.einsum('fij,jq->fqi', jacobians, quad_points)
    return phys_points

def compute_adjacency_lists(faces):
    """
    Compute lists of adjacent element pairs (non-JIT, uses numpy).

    This must be called outside JIT because it produces variable-length arrays.

    Args:
        faces: [F, 3] face connectivity (can be JAX or numpy array)

    Returns:
        Tuple containing:
        - regular_test_indices, regular_trial_indices: indices of non-adjacent (regular) pairs
        - edge_test_indices, edge_trial_indices: indices of edge-adjacent pairs
        - vertex_test_indices, vertex_trial_indices: indices of vertex-adjacent pairs
        - edge_shared_vertices: [n_edge, 4] array of (test_v1, test_v2, trial_v1, trial_v2)
        - vertex_shared_vertices: [n_vertex, 2] array of (test_v, trial_v)
    """

    faces_np = np.asarray(faces)

    # Compute shared vertex count matrix
    faces_i = faces_np[:, None, :]  # [F, 1, 3]
    faces_j = faces_np[None, :, :]  # [1, F, 3]
    matches = faces_i[:, :, :, None] == faces_j[:, :, None, :]  # [F, F, 3, 3]
    shared_count = np.sum(np.any(matches, axis=3), axis=2)  # [F, F]

    # Find regular pairs (0 shared vertices - non-adjacent)
    regular_test_indices, regular_trial_indices = np.where(shared_count == 0)

    # Find edge-adjacent pairs (2 shared vertices)
    edge_test_indices, edge_trial_indices = np.where(shared_count == 2)

    # Find vertex-adjacent pairs (1 shared vertex)
    vertex_test_indices, vertex_trial_indices = np.where(shared_count == 1)

    # For edge pairs, find which vertices are shared
    n_edge = len(edge_test_indices)
    edge_shared_vertices = np.zeros((n_edge, 4), dtype=np.int32)

    for idx in range(n_edge):
        ti, tri = edge_test_indices[idx], edge_trial_indices[idx]
        test_face = faces_np[ti]
        trial_face = faces_np[tri]

        # Find shared vertices
        test_shared = []
        trial_shared = []
        for tv in range(3):
            for trv in range(3):
                if test_face[tv] == trial_face[trv]:
                    test_shared.append(tv)
                    trial_shared.append(trv)

        edge_shared_vertices[idx] = [test_shared[0], test_shared[1],
                                      trial_shared[0], trial_shared[1]]

    # For vertex pairs, find which vertex is shared
    n_vertex = len(vertex_test_indices)
    vertex_shared_vertices = np.zeros((n_vertex, 2), dtype=np.int32)

    for idx in range(n_vertex):
        ti, tri = vertex_test_indices[idx], vertex_trial_indices[idx]
        test_face = faces_np[ti]
        trial_face = faces_np[tri]

        for tv in range(3):
            for trv in range(3):
                if test_face[tv] == trial_face[trv]:
                    vertex_shared_vertices[idx] = [tv, trv]
                    break
            else:
                continue
            break

    return (jnp.array(regular_test_indices), jnp.array(regular_trial_indices),
            jnp.array(edge_test_indices), jnp.array(edge_trial_indices),
            jnp.array(vertex_test_indices), jnp.array(vertex_trial_indices),
            jnp.array(edge_shared_vertices), jnp.array(vertex_shared_vertices))