import jax.numpy as jnp
from jax import jit
from functools import partial


def _eval_basis(space, quad_pts):
    """Evaluate basis functions [n_local, N] at reference quadrature points [2, N].

    Called only inside JIT-compiled functions with 'space' as a static arg,
    so the if/else is resolved at compile time — no runtime overhead.
    """
    if space == 'P1':
        return jnp.stack([
            1.0 - quad_pts[0] - quad_pts[1],
            quad_pts[0],
            quad_pts[1],
        ], axis=0)  # [3, N]
    else:  # DP0
        return jnp.ones((1, quad_pts.shape[1]))  # [1, N]

#%% Reference Coordinate Remapping Functions
# These transform quadrature points when the shared vertex/edge is not at
# the standard position assumed by the Duffy rules.

@jit
def remap_points_shared_vertex(points, vertex_id):
    """
    Remap triangle reference points for vertex adjacency.

    By default the Duffy rules assume triangles meet at vertex 0.
    This transforms the points based on which vertex is actually shared.

    Args:
        points: [2, N] reference coordinates
        vertex_id: 0, 1, or 2 - which vertex is shared

    Returns:
        new_points: [2, N] remapped reference coordinates
    """
    xi = points[0, :]
    eta = points[1, :]

    # Vertex 0: no change (identity)
    points_v0 = points

    # Vertex 1: rotate so vertex 1 maps to vertex 0 position
    # New coordinates: (1 - xi - eta, eta)
    points_v1 = jnp.stack([1.0 - xi - eta, eta], axis=0)

    # Vertex 2: rotate so vertex 2 maps to vertex 0 position
    # New coordinates: (xi, 1 - xi - eta)
    points_v2 = jnp.stack([xi, 1.0 - xi - eta], axis=0)

    # Select based on vertex_id using nested where
    return jnp.where(
        vertex_id == 0,
        points_v0,
        jnp.where(vertex_id == 1, points_v1, points_v2)
    )


@jit
def remap_points_shared_edge(points, shared_vertex1, shared_vertex2):
    """
    Remap triangle reference points for edge adjacency.

    By default the Duffy rules assume triangles meet at edge 0-1.
    This transforms the points based on which edge is actually shared.

    Args:
        points: [2, N] reference coordinates
        shared_vertex1: first vertex of shared edge (0, 1, or 2)
        shared_vertex2: second vertex of shared edge (0, 1, or 2)

    Returns:
        new_points: [2, N] remapped reference coordinates
    """
    # Reference triangle vertices in 2D reference space
    # v0 = (0, 0), v1 = (1, 0), v2 = (0, 1)
    ref_verts = jnp.array([
        [0.0, 1.0, 0.0],  # xi coordinates of v0, v1, v2
        [0.0, 0.0, 1.0]   # eta coordinates of v0, v1, v2
    ])

    # Build new vertex ordering: shared edge vertices first, then the third
    third_vertex = 3 - shared_vertex1 - shared_vertex2

    # New vertex positions in reference coords
    new_v0 = ref_verts[:, shared_vertex1]
    new_v1 = ref_verts[:, shared_vertex2]
    new_v2 = ref_verts[:, third_vertex]

    # Transformation matrix A and offset b such that:
    # new_point = A @ old_point + b
    # where old_point is in [0,1]^2 reference and new_point is remapped
    #
    # The mapping is: x = v0 + xi*(v1-v0) + eta*(v2-v0)
    # So A = [v1-v0, v2-v0] and b = v0

    A_col0 = new_v1 - new_v0  # [2]
    A_col1 = new_v2 - new_v0  # [2]
    A = jnp.stack([A_col0, A_col1], axis=1)  # [2, 2]
    b = new_v0  # [2]

    # Apply transformation: new_points = A @ points + b
    new_points = A @ points + b[:, None]

    return new_points

@jit
def _precompute_remapped_vertex_rules(base_points, base_weights):
    """
    Pre-compute all 3 vertex-remapped versions of quadrature points.

    Args:
        base_points: [2, N] base quadrature points
        base_weights: [N] quadrature weights (same for all remappings)

    Returns:
        all_points: [3, 2, N] - points for vertex_id 0, 1, 2
        weights: [N] - unchanged weights
    """
    points_v0 = remap_points_shared_vertex(base_points, 0)
    points_v1 = remap_points_shared_vertex(base_points, 1)
    points_v2 = remap_points_shared_vertex(base_points, 2)

    return jnp.stack([points_v0, points_v1, points_v2], axis=0), base_weights

@jit
def _precompute_remapped_edge_rules(base_points, base_weights):
    """
    Pre-compute all 6 edge-remapped versions of quadrature points.

    The 6 configurations correspond to shared edges:
    0: (0, 1), 1: (1, 0), 2: (1, 2), 3: (2, 1), 4: (0, 2), 5: (2, 0)

    Args:
        base_points: [2, N] base quadrature points
        base_weights: [N] quadrature weights (same for all remappings)

    Returns:
        all_points: [6, 2, N] - points for each edge configuration
        weights: [N] - unchanged weights
    """
    # Edge configurations: (v1, v2) pairs
    edge_configs = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]

    remapped = []
    for v1, v2 in edge_configs:
        remapped.append(remap_points_shared_edge(base_points, v1, v2))

    return jnp.stack(remapped, axis=0), base_weights

@jit
def _get_edge_remap_index(shared_v1, shared_v2):
    """
    Get the index into the pre-computed edge rules based on shared vertices.

    Edge configurations indexed as:
    0: (0, 1), 1: (1, 0), 2: (1, 2), 3: (2, 1), 4: (0, 2), 5: (2, 0)
    """
    # Lookup table: edge_index[v1, v2] gives the remap index
    # Diagonal entries are invalid (-1) but shouldn't be accessed
    edge_index_table = jnp.array([
        [-1,  0,  4],  # v1=0: (0,1)->0, (0,2)->4
        [ 1, -1,  2],  # v1=1: (1,0)->1, (1,2)->2
        [ 5,  3, -1]   # v1=2: (2,0)->5, (2,1)->3
    ], dtype=jnp.int32)

    return edge_index_table[shared_v1, shared_v2]


#%% Precompute Gauss-Legendre rules and quadrature
_GAUSS_LEGENDRE_RULES = {}

def _precompute_gauss_rules():
    import numpy as np
    for order in range(1, 11):
        points_std, weights_std = np.polynomial.legendre.leggauss(order)
        # Transform from [-1, 1] to [0, 1]
        points = 0.5 * (points_std + 1.0)
        weights = 0.5 * weights_std
        _GAUSS_LEGENDRE_RULES[order] = (
            jnp.array(points),
            jnp.array(weights)
        )

_precompute_gauss_rules()

def gauss_legendre_1d(order):
    """Gauss-Legendre quadrature on [0, 1] - JIT compatible."""
    if order not in _GAUSS_LEGENDRE_RULES:
        raise ValueError(f"Order {order} not precomputed. Max is 10.")
    return _GAUSS_LEGENDRE_RULES[order]

@partial(jit, static_argnames=['order'])
def singular_quadrature_coincident(order):
    """
    Generate Duffy quadrature rule for coincident triangles.
    
    Args:
        order: Number of Gauss points per dimension
        
    Returns:
        points_test: [2, N] reference coordinates on test triangle
        points_trial: [2, N] reference coordinates on trial triangle  
        weights: [N] quadrature weights (includes Duffy Jacobian)
        
    where N = 6 * order^4
    """
    # 1D Gauss rule on [0, 1]
    x1d, w1d = gauss_legendre_1d(order)
    n1d = order
    
    # Build 2D tensor product rule on [0,1]^2
    # Shape: [order^2] for each
    xi_2d = jnp.tile(x1d, n1d)                          # [Q^2]
    eta_2d = jnp.repeat(x1d, n1d)                       # [Q^2]
    w_2d = jnp.outer(w1d, w1d).ravel()                  # [Q^2]
    
    n2d = n1d * n1d
    
    # Build 4D tensor product: (xsi, eta1) x (eta2, eta3)
    # Each has n2d points, giving n2d^2 = order^4 combinations
    
    # Coordinates in 4D parameter space
    xsi = jnp.tile(xi_2d, n2d)                          # [Q^4]
    eta1 = jnp.tile(eta_2d, n2d)                        # [Q^4]
    eta2 = jnp.repeat(xi_2d, n2d)                       # [Q^4]
    eta3 = jnp.repeat(eta_2d, n2d)                      # [Q^4]
    
    # 4D tensor weight
    w_4d = jnp.outer(w_2d, w_2d).ravel()                # [Q^4]
    
    # Precompute common terms
    eta12 = eta1 * eta2
    eta123 = eta1 * eta2 * eta3
    
    # Duffy Jacobian (absorbs the singularity)
    jacobian = xsi**3 * eta1**2 * eta2
    base_weight = w_4d * jacobian
    
    # 6 regions for coincident case
    # Each region maps [0,1]^4 -> (test triangle) x (trial triangle)
    
    # Region 1
    test_xi_1 = xsi
    test_eta_1 = xsi * (1.0 - eta1 + eta12)
    trial_xi_1 = xsi * (1.0 - eta123)
    trial_eta_1 = xsi * (1.0 - eta1)
    
    # Region 2 (symmetric to region 1)
    test_xi_2 = xsi * (1.0 - eta123)
    test_eta_2 = xsi * (1.0 - eta1)
    trial_xi_2 = xsi
    trial_eta_2 = xsi * (1.0 - eta1 + eta12)
    
    # Region 3
    test_xi_3 = xsi
    test_eta_3 = xsi * (eta1 - eta12 + eta123)
    trial_xi_3 = xsi * (1.0 - eta12)
    trial_eta_3 = xsi * (eta1 - eta12)
    
    # Region 4 (symmetric to region 3)
    test_xi_4 = xsi * (1.0 - eta12)
    test_eta_4 = xsi * (eta1 - eta12)
    trial_xi_4 = xsi
    trial_eta_4 = xsi * (eta1 - eta12 + eta123)
    
    # Region 5
    test_xi_5 = xsi * (1.0 - eta123)
    test_eta_5 = xsi * (eta1 - eta123)
    trial_xi_5 = xsi
    trial_eta_5 = xsi * (eta1 - eta12)
    
    # Region 6 (symmetric to region 5)
    test_xi_6 = xsi
    test_eta_6 = xsi * (eta1 - eta12)
    trial_xi_6 = xsi * (1.0 - eta123)
    trial_eta_6 = xsi * (eta1 - eta123)
    
    # Stack all 6 regions: [6, Q^4] -> [6*Q^4]
    test_xi = jnp.concatenate([test_xi_1, test_xi_2, test_xi_3, 
                                test_xi_4, test_xi_5, test_xi_6])
    test_eta = jnp.concatenate([test_eta_1, test_eta_2, test_eta_3,
                                 test_eta_4, test_eta_5, test_eta_6])
    trial_xi = jnp.concatenate([trial_xi_1, trial_xi_2, trial_xi_3,
                                 trial_xi_4, trial_xi_5, trial_xi_6])
    trial_eta = jnp.concatenate([trial_eta_1, trial_eta_2, trial_eta_3,
                                  trial_eta_4, trial_eta_5, trial_eta_6])
    
    # All 6 regions have the same weight
    weights = jnp.tile(base_weight, 6)
    
    # Convert to Bempp's reference triangle convention
    # Reference triangle: (0,0), (1,0), (0,1)
    points_test = jnp.stack([test_xi - test_eta, test_eta], axis=0)    # [2, N]
    points_trial = jnp.stack([trial_xi - trial_eta, trial_eta], axis=0)  # [2, N]
    
    return points_test, points_trial, weights

@partial(jit, static_argnames=['order'])
def singular_quadrature_edge_adjacent(order):
    """
    Generate Duffy quadrature rule for edge-adjacent triangles.

    Edge-adjacent means the triangles share exactly 2 vertices (a common edge).

    Args:
        order: Number of Gauss points per dimension

    Returns:
        points_test: [2, N] reference coordinates on test triangle
        points_trial: [2, N] reference coordinates on trial triangle
        weights: [N] quadrature weights (includes Duffy Jacobian)

    where N = 5 * order^4
    """
    # 1D Gauss rule on [0, 1]
    x1d, w1d = gauss_legendre_1d(order)
    n1d = order

    # Build 2D tensor product rule on [0,1]^2
    xi_2d = jnp.tile(x1d, n1d)
    eta_2d = jnp.repeat(x1d, n1d)
    w_2d = jnp.outer(w1d, w1d).ravel()

    n2d = n1d * n1d

    # Build 4D tensor product
    xsi = jnp.tile(xi_2d, n2d)
    eta1 = jnp.tile(eta_2d, n2d)
    eta2 = jnp.repeat(xi_2d, n2d)
    eta3 = jnp.repeat(eta_2d, n2d)

    w_4d = jnp.outer(w_2d, w_2d).ravel()

    # Precompute common terms
    eta12 = eta1 * eta2
    eta123 = eta1 * eta2 * eta3

    # Base Duffy Jacobian for edge-adjacent
    jacobian = xsi**3 * eta1**2
    base_weight = w_4d * jacobian

    # 5 regions for edge-adjacent case

    # Region 1
    test_xi_1 = xsi
    test_eta_1 = xsi * eta1 * eta3
    trial_xi_1 = xsi * (1.0 - eta12)
    trial_eta_1 = xsi * eta1 * (1.0 - eta2)
    weight_1 = base_weight

    # Region 2
    test_xi_2 = xsi
    test_eta_2 = xsi * eta1
    trial_xi_2 = xsi * (1.0 - eta123)
    trial_eta_2 = xsi * eta12 * (1.0 - eta3)
    weight_2 = base_weight * eta2

    # Region 3
    test_xi_3 = xsi * (1.0 - eta12)
    test_eta_3 = xsi * eta1 * (1.0 - eta2)
    trial_xi_3 = xsi
    trial_eta_3 = xsi * eta123
    weight_3 = base_weight * eta2

    # Region 4
    test_xi_4 = xsi * (1.0 - eta123)
    test_eta_4 = xsi * eta12 * (1.0 - eta3)
    trial_xi_4 = xsi
    trial_eta_4 = xsi * eta1
    weight_4 = base_weight * eta2

    # Region 5
    test_xi_5 = xsi * (1.0 - eta123)
    test_eta_5 = xsi * eta1 * (1.0 - eta2 * eta3)
    trial_xi_5 = xsi
    trial_eta_5 = xsi * eta12
    weight_5 = base_weight * eta2

    # Stack all 5 regions: [5, Q^4] -> [5*Q^4]
    test_xi = jnp.concatenate([test_xi_1, test_xi_2, test_xi_3, test_xi_4, test_xi_5])
    test_eta = jnp.concatenate([test_eta_1, test_eta_2, test_eta_3, test_eta_4, test_eta_5])
    trial_xi = jnp.concatenate([trial_xi_1, trial_xi_2, trial_xi_3, trial_xi_4, trial_xi_5])
    trial_eta = jnp.concatenate([trial_eta_1, trial_eta_2, trial_eta_3, trial_eta_4, trial_eta_5])
    weights = jnp.concatenate([weight_1, weight_2, weight_3, weight_4, weight_5])

    # Convert to Bempp's reference triangle convention
    points_test = jnp.stack([test_xi - test_eta, test_eta], axis=0)
    points_trial = jnp.stack([trial_xi - trial_eta, trial_eta], axis=0)

    return points_test, points_trial, weights

@partial(jit, static_argnames=['order'])
def singular_quadrature_vertex_adjacent(order):
    """
    Generate Duffy quadrature rule for vertex-adjacent triangles.

    Vertex-adjacent means the triangles share exactly 1 vertex.

    Args:
        order: Number of Gauss points per dimension

    Returns:
        points_test: [2, N] reference coordinates on test triangle
        points_trial: [2, N] reference coordinates on trial triangle
        weights: [N] quadrature weights (includes Duffy Jacobian)

    where N = 2 * order^4
    """
    # 1D Gauss rule on [0, 1]
    x1d, w1d = gauss_legendre_1d(order)
    n1d = order

    # Build 2D tensor product rule on [0,1]^2
    xi_2d = jnp.tile(x1d, n1d)
    eta_2d = jnp.repeat(x1d, n1d)
    w_2d = jnp.outer(w1d, w1d).ravel()

    n2d = n1d * n1d

    # Build 4D tensor product
    xsi = jnp.tile(xi_2d, n2d)
    eta1 = jnp.tile(eta_2d, n2d)
    eta2 = jnp.repeat(xi_2d, n2d)
    eta3 = jnp.repeat(eta_2d, n2d)

    w_4d = jnp.outer(w_2d, w_2d).ravel()

    # Precompute common terms
    eta23 = eta2 * eta3

    # Base Duffy Jacobian for vertex-adjacent
    jacobian = xsi**3 * eta2
    base_weight = w_4d * jacobian

    # 2 regions for vertex-adjacent case

    # Region 1
    test_xi_1 = xsi
    test_eta_1 = xsi * eta1
    trial_xi_1 = xsi * eta2
    trial_eta_1 = xsi * eta23
    weight_1 = base_weight

    # Region 2
    test_xi_2 = xsi * eta2
    test_eta_2 = xsi * eta23
    trial_xi_2 = xsi
    trial_eta_2 = xsi * eta1
    weight_2 = base_weight

    # Stack both regions: [2, Q^4] -> [2*Q^4]
    test_xi = jnp.concatenate([test_xi_1, test_xi_2])
    test_eta = jnp.concatenate([test_eta_1, test_eta_2])
    trial_xi = jnp.concatenate([trial_xi_1, trial_xi_2])
    trial_eta = jnp.concatenate([trial_eta_1, trial_eta_2])
    weights = jnp.concatenate([weight_1, weight_2])

    # Convert to Bempp's reference triangle convention
    points_test = jnp.stack([test_xi - test_eta, test_eta], axis=0)
    points_trial = jnp.stack([trial_xi - trial_eta, trial_eta], axis=0)

    return points_test, points_trial, weights


#%% Pre-computed Remapped Quadrature Rules
# These are computed once at module load time for efficiency.

_REMAPPED_RULES = {}

def _precompute_all_remapped_rules():
    """Pre-compute all remapped quadrature rules for orders 1-10."""
    for order in range(1, 11):
        # Edge-adjacent remapped rules
        base_test, base_trial, weights_edge = singular_quadrature_edge_adjacent(order)
        test_edge_remapped, _ = _precompute_remapped_edge_rules(base_test, weights_edge)
        trial_edge_remapped, _ = _precompute_remapped_edge_rules(base_trial, weights_edge)

        # Vertex-adjacent remapped rules
        base_test_v, base_trial_v, weights_vertex = singular_quadrature_vertex_adjacent(order)
        test_vertex_remapped, _ = _precompute_remapped_vertex_rules(base_test_v, weights_vertex)
        trial_vertex_remapped, _ = _precompute_remapped_vertex_rules(base_trial_v, weights_vertex)

        _REMAPPED_RULES[order] = {
            'edge_adjacent': {
                'test': test_edge_remapped,      # [6, 2, N]
                'trial': trial_edge_remapped,    # [6, 2, N]
                'weights': weights_edge          # [N]
            },
            'vertex_adjacent': {
                'test': test_vertex_remapped,    # [3, 2, N]
                'trial': trial_vertex_remapped,  # [3, 2, N]
                'weights': weights_vertex        # [N]
            }
        }

# Precompute at module load
_precompute_all_remapped_rules()


def get_remapped_edge_adjacent_rule(order, test_shared_v1, test_shared_v2,
                                     trial_shared_v1, trial_shared_v2):
    """
    Get the remapped edge-adjacent quadrature rule for given shared edge configuration.

    Args:
        order: quadrature order
        test_shared_v1, test_shared_v2: local vertex indices of shared edge in test element
        trial_shared_v1, trial_shared_v2: local vertex indices of shared edge in trial element

    Returns:
        points_test: [2, N] remapped test quadrature points
        points_trial: [2, N] remapped trial quadrature points
        weights: [N] quadrature weights
    """
    rules = _REMAPPED_RULES[order]['edge_adjacent']
    test_idx = _get_edge_remap_index(test_shared_v1, test_shared_v2)
    trial_idx = _get_edge_remap_index(trial_shared_v1, trial_shared_v2)

    return rules['test'][test_idx], rules['trial'][trial_idx], rules['weights']


def get_remapped_vertex_adjacent_rule(order, test_shared_vertex, trial_shared_vertex):
    """
    Get the remapped vertex-adjacent quadrature rule for given shared vertex configuration.

    Args:
        order: quadrature order
        test_shared_vertex: local vertex index of shared vertex in test element (0, 1, or 2)
        trial_shared_vertex: local vertex index of shared vertex in trial element (0, 1, or 2)

    Returns:
        points_test: [2, N] remapped test quadrature points
        points_trial: [2, N] remapped trial quadrature points
        weights: [N] quadrature weights
    """
    rules = _REMAPPED_RULES[order]['vertex_adjacent']

    return rules['test'][test_shared_vertex], rules['trial'][trial_shared_vertex], rules['weights']


#%% Coincident local matrix computations

@partial(jit, static_argnames=['order', 'space'])
def compute_coincident_double_layer_matrix(
    vertices,    # [3, 3] - three vertices of the triangle (vertex i is vertices[i])
    normal,      # [3] - element normal
    k0,          # wavenumber (real)
    order=4,     # quadrature order per dimension
    space='P1',  # function space identifier
    ):
    """
    Compute local 3x3 double layer matrix for a coincident (self) element.
    
    Uses Duffy transformation to handle the singularity.
    """
    # Get singular quadrature rule
    # points_test, points_trial: [2, N] in reference coordinates
    # weights: [N] includes Duffy Jacobian
    points_test, points_trial, weights = singular_quadrature_coincident(order)

    # Map reference points to physical coordinates
    # Physical point = v0 + xi * (v1 - v0) + eta * (v2 - v0)
    v0 = vertices[0]  # [3]
    v1 = vertices[1]  # [3]
    v2 = vertices[2]  # [3]

    e1 = v1 - v0  # [3]
    e2 = v2 - v0  # [3]

    # Integration element (area scaling)
    int_elem = jnp.linalg.norm(jnp.cross(e1, e2)) / 2.0
    
    # Physical test points: [N, 3]
    phys_test = v0[None, :] + points_test[0, :, None] * e1[None, :] + points_test[1, :, None] * e2[None, :]
    
    # Physical trial points: [N, 3]
    phys_trial = v0[None, :] + points_trial[0, :, None] * e1[None, :] + points_trial[1, :, None] * e2[None, :]
    
    # Evaluate kernel at all quadrature point pairs
    # For coincident elements, we have paired points (not all combinations)
    # diff[i] = phys_trial[i] - phys_test[i]
    diff = phys_trial - phys_test  # [N, 3]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))  # [N]
    
    # Double layer kernel: (d/dn_y) G(x,y) where G = exp(ik|x-y|) / (4π|x-y|)
    # = (exp(ikr) / 4πr) * (ikr - 1) / r^2 * (y-x)·n_y
    # But for coincident, normal is the same for test and trial
    
    # Dot product of diff with normal
    diff_dot_n = jnp.sum(diff * normal[None, :], axis=1)  # [N]
    
    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    
    # Laplace gradient part: (y-x)·n / |x-y|^3
    laplace_grad = diff_dot_n * M_INV_4PI / (dist ** 3)
    
    # Helmholtz kernel
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = laplace_grad * phase * (-1 + 1j * k0 * dist)  # [N]
    
    # Full quadrature weight including integration elements
    # Factor of 4 because int_elem appears twice (test and trial) and 
    # reference triangle has area 0.5 (so 0.5 * 0.5 = 0.25, need to scale by 4)
    full_weights = weights * int_elem * int_elem * 4.0
    
    # Weighted kernel values
    weighted_kernel = kernel_vals * full_weights  # [N]
    
    basis_test  = _eval_basis(space, points_test)   # [n_local, N]
    basis_trial = _eval_basis(space, points_trial)  # [n_local, N]
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)

@partial(jit, static_argnames=['order', 'space'])
def compute_coincident_hypersingular_matrix(
    vertices,    # [3, 3] - three vertices of the triangle (vertex i is vertices[i])
    normal,      # [3] - element normal
    k0,          # wavenumber (real)
    order=4,     # quadrature order per dimension
    space='P1',  # function space identifier
    ):
    """
    Compute local 3x3 hypersingular matrix for a coincident (self) element.
    
    Uses Duffy transformation to handle the singularity.
    
    The hypersingular operator is:
    W[i,j] = ∫∫ (curl φ_i · curl φ_j * G - k² * (n·n) * φ_i * φ_j * G) dS_x dS_y
    
    where G is the adjoint double layer kernel.
    """
    # Get singular quadrature rule
    points_test, points_trial, weights = singular_quadrature_coincident(order)

    # Geometry setup
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    
    e1 = v1 - v0
    e2 = v2 - v0
    
    # Jacobian matrix [3, 2]
    jacobian = jnp.stack([e1, e2], axis=1)
    
    # Integration element
    cross = jnp.cross(e1, e2)
    int_elem = jnp.linalg.norm(cross) / 2.0
    
    # Curl-curl term: present only for P1 (constant basis has zero surface gradient)
    if space == 'P1':
        JTJ = jacobian.T @ jacobian
        JTJ_inv = jnp.linalg.inv(JTJ)
        jac_inv_trans = jacobian @ JTJ_inv  # [3, 2]
        ref_gradients = jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        surface_gradients = ref_gradients @ jac_inv_trans.T
        surface_curls = jnp.cross(normal[None, :], surface_gradients)  # [3, 3]
        curl_product = surface_curls @ surface_curls.T                  # [3, 3]

    # Normal product (coincident: same element)
    normal_prod = 1.0

    # Physical points
    phys_test  = v0[None, :] + points_test[0,  :, None] * e1[None, :] + points_test[1,  :, None] * e2[None, :]
    phys_trial = v0[None, :] + points_trial[0, :, None] * e1[None, :] + points_trial[1, :, None] * e2[None, :]

    # Single layer kernel
    diff = phys_test - phys_trial
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = phase * M_INV_4PI / dist

    full_weights = weights * int_elem * int_elem * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)   # [n_local, N]
    basis_trial = _eval_basis(space, points_trial)  # [n_local, N]
    mass_term = jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)

    if space == 'P1':
        curl_term = jnp.sum(weighted_kernel) * curl_product
    else:  # DP0: surface curl of constant basis is zero
        curl_term = jnp.zeros((1, 1))

    return curl_term - k0**2 * normal_prod * mass_term

@partial(jit, static_argnames=['order', 'space'])
def compute_edge_adjacent_double_layer_matrix(
    test_vertices,   # [3, 3] - test triangle vertices
    trial_vertices,  # [3, 3] - trial triangle vertices
    test_normal,     # [3] - test element normal
    trial_normal,    # [3] - trial element normal
    k0,              # wavenumber
    test_shared_v1,  # local index of first shared vertex in test element
    test_shared_v2,  # local index of second shared vertex in test element
    trial_shared_v1, # local index of first shared vertex in trial element
    trial_shared_v2, # local index of second shared vertex in trial element
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 double layer matrix for edge-adjacent elements.

    Uses 5-region Duffy transformation for edge-adjacent singularities.
    The quadrature points are remapped based on which edge is shared.
    """
    # Get remapped edge-adjacent quadrature rule
    points_test, points_trial, weights = get_remapped_edge_adjacent_rule(
        order, test_shared_v1, test_shared_v2, trial_shared_v1, trial_shared_v2
    )

    # Test geometry
    v0_test = test_vertices[0]
    e1_test = test_vertices[1] - v0_test
    e2_test = test_vertices[2] - v0_test
    cross_test = jnp.cross(e1_test, e2_test)
    int_elem_test = jnp.linalg.norm(cross_test) / 2.0

    # Trial geometry
    v0_trial = trial_vertices[0]
    e1_trial = trial_vertices[1] - v0_trial
    e2_trial = trial_vertices[2] - v0_trial
    cross_trial = jnp.cross(e1_trial, e2_trial)
    int_elem_trial = jnp.linalg.norm(cross_trial) / 2.0

    # Physical points
    phys_test = v0_test[None, :] + points_test[0, :, None] * e1_test[None, :] + points_test[1, :, None] * e2_test[None, :]
    phys_trial = v0_trial[None, :] + points_trial[0, :, None] * e1_trial[None, :] + points_trial[1, :, None] * e2_trial[None, :]

    # Kernel evaluation
    diff = phys_trial - phys_test
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    diff_dot_n = jnp.sum(diff * trial_normal[None, :], axis=1)

    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    laplace_grad = diff_dot_n * M_INV_4PI / (dist ** 3)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = laplace_grad * phase * (-1 + 1j * k0 * dist)

    # Full quadrature weights
    full_weights = weights * int_elem_test * int_elem_trial * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)

@partial(jit, static_argnames=['order', 'space'])
def compute_vertex_adjacent_double_layer_matrix(
    test_vertices,       # [3, 3] - test triangle vertices
    trial_vertices,      # [3, 3] - trial triangle vertices
    test_normal,         # [3] - test element normal
    trial_normal,        # [3] - trial element normal
    k0,                  # wavenumber
    test_shared_vertex,  # local index of shared vertex in test element
    trial_shared_vertex, # local index of shared vertex in trial element
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 double layer matrix for vertex-adjacent elements.

    Uses 2-region Duffy transformation for vertex-adjacent singularities.
    Quadrature points are remapped based on which vertex is shared.
    """
    # Get remapped vertex-adjacent quadrature rule
    points_test, points_trial, weights = get_remapped_vertex_adjacent_rule(
        order, test_shared_vertex, trial_shared_vertex
    )

    # Test geometry
    v0_test = test_vertices[0]
    e1_test = test_vertices[1] - v0_test
    e2_test = test_vertices[2] - v0_test
    cross_test = jnp.cross(e1_test, e2_test)
    int_elem_test = jnp.linalg.norm(cross_test) / 2.0

    # Trial geometry
    v0_trial = trial_vertices[0]
    e1_trial = trial_vertices[1] - v0_trial
    e2_trial = trial_vertices[2] - v0_trial
    cross_trial = jnp.cross(e1_trial, e2_trial)
    int_elem_trial = jnp.linalg.norm(cross_trial) / 2.0

    # Physical points
    phys_test = v0_test[None, :] + points_test[0, :, None] * e1_test[None, :] + points_test[1, :, None] * e2_test[None, :]
    phys_trial = v0_trial[None, :] + points_trial[0, :, None] * e1_trial[None, :] + points_trial[1, :, None] * e2_trial[None, :]

    # Kernel evaluation
    diff = phys_trial - phys_test
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    diff_dot_n = jnp.sum(diff * trial_normal[None, :], axis=1)

    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    laplace_grad = diff_dot_n * M_INV_4PI / (dist ** 3)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = laplace_grad * phase * (-1 + 1j * k0 * dist)

    # Full quadrature weights
    full_weights = weights * int_elem_test * int_elem_trial * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)


#%% Adjoint Double Layer Singular Matrices

@partial(jit, static_argnames=['order', 'space'])
def compute_coincident_adjoint_double_layer_matrix(
    vertices,   # [3, 3] - three vertices of the triangle
    normal,     # [3] - element normal
    k0,         # wavenumber
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 adjoint double layer matrix for a coincident (self) element.

    K'[i,j] = ∫∫ ∂G(x,y)/∂n(x) φ_i(x) φ_j(y) dS_x dS_y

    Kernel: (x-y)·n_x / (4π|x-y|³) * exp(ik|x-y|) * (-1 + ik|x-y|)

    For coincident elements n_x = n_y = n, so the kernel is the negative of the
    double layer kernel (diff direction reversed, same normal).
    """
    points_test, points_trial, weights = singular_quadrature_coincident(order)

    v0 = vertices[0]
    e1 = vertices[1] - v0
    e2 = vertices[2] - v0
    int_elem = jnp.linalg.norm(jnp.cross(e1, e2)) / 2.0

    phys_test  = v0[None, :] + points_test[0,  :, None] * e1[None, :] + points_test[1,  :, None] * e2[None, :]
    phys_trial = v0[None, :] + points_trial[0, :, None] * e1[None, :] + points_trial[1, :, None] * e2[None, :]

    diff = phys_test - phys_trial  # x - y  (negated vs DL)
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    diff_dot_n = jnp.sum(diff * normal[None, :], axis=1)  # (x-y)·n_x

    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    laplace_grad = diff_dot_n * M_INV_4PI / (dist ** 3)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = laplace_grad * phase * (-1 + 1j * k0 * dist)

    full_weights = weights * int_elem * int_elem * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)


@partial(jit, static_argnames=['order', 'space'])
def compute_edge_adjacent_adjoint_double_layer_matrix(
    test_vertices,   # [3, 3]
    trial_vertices,  # [3, 3]
    test_normal,     # [3]
    trial_normal,    # [3]  (unused by kernel, kept for consistent API)
    k0,
    test_shared_v1,
    test_shared_v2,
    trial_shared_v1,
    trial_shared_v2,
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 adjoint double layer matrix for edge-adjacent elements.

    Uses the same 5-region Duffy transform as the double layer, but with
    diff = x - y and test normal instead of trial normal.
    """
    points_test, points_trial, weights = get_remapped_edge_adjacent_rule(
        order, test_shared_v1, test_shared_v2, trial_shared_v1, trial_shared_v2)

    v0_test  = test_vertices[0]
    e1_test  = test_vertices[1] - v0_test
    e2_test  = test_vertices[2] - v0_test
    int_elem_test = jnp.linalg.norm(jnp.cross(e1_test, e2_test)) / 2.0

    v0_trial  = trial_vertices[0]
    e1_trial  = trial_vertices[1] - v0_trial
    e2_trial  = trial_vertices[2] - v0_trial
    int_elem_trial = jnp.linalg.norm(jnp.cross(e1_trial, e2_trial)) / 2.0

    phys_test  = v0_test[None,  :] + points_test[0,  :, None] * e1_test[None,  :] + points_test[1,  :, None] * e2_test[None,  :]
    phys_trial = v0_trial[None, :] + points_trial[0, :, None] * e1_trial[None, :] + points_trial[1, :, None] * e2_trial[None, :]

    diff = phys_test - phys_trial  # x - y
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    diff_dot_n = jnp.sum(diff * test_normal[None, :], axis=1)  # (x-y)·n_x

    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    laplace_grad = diff_dot_n * M_INV_4PI / (dist ** 3)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = laplace_grad * phase * (-1 + 1j * k0 * dist)

    full_weights = weights * int_elem_test * int_elem_trial * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)


@partial(jit, static_argnames=['order', 'space'])
def compute_vertex_adjacent_adjoint_double_layer_matrix(
    test_vertices,       # [3, 3]
    trial_vertices,      # [3, 3]
    test_normal,         # [3]
    trial_normal,        # [3]  (unused by kernel, kept for consistent API)
    k0,
    test_shared_vertex,
    trial_shared_vertex,
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 adjoint double layer matrix for vertex-adjacent elements.

    Uses the same 2-region Duffy transform as the double layer, but with
    diff = x - y and test normal instead of trial normal.
    """
    points_test, points_trial, weights = get_remapped_vertex_adjacent_rule(
        order, test_shared_vertex, trial_shared_vertex)

    v0_test  = test_vertices[0]
    e1_test  = test_vertices[1] - v0_test
    e2_test  = test_vertices[2] - v0_test
    int_elem_test = jnp.linalg.norm(jnp.cross(e1_test, e2_test)) / 2.0

    v0_trial  = trial_vertices[0]
    e1_trial  = trial_vertices[1] - v0_trial
    e2_trial  = trial_vertices[2] - v0_trial
    int_elem_trial = jnp.linalg.norm(jnp.cross(e1_trial, e2_trial)) / 2.0

    phys_test  = v0_test[None,  :] + points_test[0,  :, None] * e1_test[None,  :] + points_test[1,  :, None] * e2_test[None,  :]
    phys_trial = v0_trial[None, :] + points_trial[0, :, None] * e1_trial[None, :] + points_trial[1, :, None] * e2_trial[None, :]

    diff = phys_test - phys_trial  # x - y
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    diff_dot_n = jnp.sum(diff * test_normal[None, :], axis=1)  # (x-y)·n_x

    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    laplace_grad = diff_dot_n * M_INV_4PI / (dist ** 3)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = laplace_grad * phase * (-1 + 1j * k0 * dist)

    full_weights = weights * int_elem_test * int_elem_trial * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)


@partial(jit, static_argnames=['order', 'space'])
def compute_edge_adjacent_hypersingular_matrix(
    test_vertices,   # [3, 3] - test triangle vertices
    trial_vertices,  # [3, 3] - trial triangle vertices
    test_normal,     # [3] - test element normal
    trial_normal,    # [3] - trial element normal
    k0,              # wavenumber
    test_shared_v1,  # local index of first shared vertex in test element
    test_shared_v2,  # local index of second shared vertex in test element
    trial_shared_v1, # local index of first shared vertex in trial element
    trial_shared_v2, # local index of second shared vertex in trial element
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 hypersingular matrix for edge-adjacent elements.

    Uses 5-region Duffy transformation for edge-adjacent singularities.
    Quadrature points are remapped based on which edge is shared.
    """
    # Get remapped edge-adjacent quadrature rule
    points_test, points_trial, weights = get_remapped_edge_adjacent_rule(
        order, test_shared_v1, test_shared_v2, trial_shared_v1, trial_shared_v2
    )

    # Test geometry
    v0_test = test_vertices[0]
    e1_test = test_vertices[1] - v0_test
    e2_test = test_vertices[2] - v0_test
    jacobian_test = jnp.stack([e1_test, e2_test], axis=1)
    cross_test = jnp.cross(e1_test, e2_test)
    int_elem_test = jnp.linalg.norm(cross_test) / 2.0

    # Trial geometry
    v0_trial = trial_vertices[0]
    e1_trial = trial_vertices[1] - v0_trial
    e2_trial = trial_vertices[2] - v0_trial
    jacobian_trial = jnp.stack([e1_trial, e2_trial], axis=1)
    cross_trial = jnp.cross(e1_trial, e2_trial)
    int_elem_trial = jnp.linalg.norm(cross_trial) / 2.0

    if space == 'P1':
        ref_gradients = jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        JTJ_test_inv = jnp.linalg.inv(jacobian_test.T @ jacobian_test)
        surface_curls_test = jnp.cross(
            test_normal[None, :],
            ref_gradients @ (jacobian_test @ JTJ_test_inv).T)
        JTJ_trial_inv = jnp.linalg.inv(jacobian_trial.T @ jacobian_trial)
        surface_curls_trial = jnp.cross(
            trial_normal[None, :],
            ref_gradients @ (jacobian_trial @ JTJ_trial_inv).T)
        curl_product = surface_curls_test @ surface_curls_trial.T  # [3, 3]

    # Physical points
    phys_test  = v0_test[None,  :] + points_test[0,  :, None] * e1_test[None,  :] + points_test[1,  :, None] * e2_test[None,  :]
    phys_trial = v0_trial[None, :] + points_trial[0, :, None] * e1_trial[None, :] + points_trial[1, :, None] * e2_trial[None, :]

    diff = phys_test - phys_trial
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = phase * M_INV_4PI / dist

    full_weights = weights * int_elem_test * int_elem_trial * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    normal_prod = jnp.dot(test_normal, trial_normal)
    mass_term = jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)

    if space == 'P1':
        curl_term = jnp.sum(weighted_kernel) * curl_product
    else:
        curl_term = jnp.zeros((1, 1))

    return curl_term - k0**2 * normal_prod * mass_term


#%% Single Layer Singular Matrices

@partial(jit, static_argnames=['order', 'space'])
def compute_coincident_single_layer_matrix(
    vertices,   # [3, 3] - three vertices of the triangle
    k0,         # wavenumber
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 single layer matrix for a coincident (self) element.

    V[i,j] = ∫∫ G(x,y) φ_i(x) φ_j(y) dS_x dS_y
    where G(x,y) = exp(ik|x-y|) / (4π|x-y|)

    Uses Duffy transformation to handle the 1/r weak singularity.
    """
    points_test, points_trial, weights = singular_quadrature_coincident(order)

    v0 = vertices[0]
    e1 = vertices[1] - v0
    e2 = vertices[2] - v0
    int_elem = jnp.linalg.norm(jnp.cross(e1, e2)) / 2.0

    phys_test  = v0[None, :] + points_test[0,  :, None] * e1[None, :] + points_test[1,  :, None] * e2[None, :]
    phys_trial = v0[None, :] + points_trial[0, :, None] * e1[None, :] + points_trial[1, :, None] * e2[None, :]

    diff = phys_trial - phys_test
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))

    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = phase * M_INV_4PI / dist

    full_weights = weights * int_elem * int_elem * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)


@partial(jit, static_argnames=['order', 'space'])
def compute_edge_adjacent_single_layer_matrix(
    test_vertices,   # [3, 3]
    trial_vertices,  # [3, 3]
    k0,
    test_shared_v1,
    test_shared_v2,
    trial_shared_v1,
    trial_shared_v2,
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 single layer matrix for edge-adjacent elements.

    Uses 5-region Duffy transformation for the 1/r weak singularity at the shared edge.
    """
    points_test, points_trial, weights = get_remapped_edge_adjacent_rule(
        order, test_shared_v1, test_shared_v2, trial_shared_v1, trial_shared_v2
    )

    v0_test  = test_vertices[0]
    e1_test  = test_vertices[1] - v0_test
    e2_test  = test_vertices[2] - v0_test
    int_elem_test  = jnp.linalg.norm(jnp.cross(e1_test,  e2_test))  / 2.0

    v0_trial = trial_vertices[0]
    e1_trial = trial_vertices[1] - v0_trial
    e2_trial = trial_vertices[2] - v0_trial
    int_elem_trial = jnp.linalg.norm(jnp.cross(e1_trial, e2_trial)) / 2.0

    phys_test  = v0_test[None,  :] + points_test[0,  :, None] * e1_test[None,  :] + points_test[1,  :, None] * e2_test[None,  :]
    phys_trial = v0_trial[None, :] + points_trial[0, :, None] * e1_trial[None, :] + points_trial[1, :, None] * e2_trial[None, :]

    diff = phys_trial - phys_test
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))

    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = phase * M_INV_4PI / dist

    full_weights = weights * int_elem_test * int_elem_trial * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)


@partial(jit, static_argnames=['order', 'space'])
def compute_vertex_adjacent_single_layer_matrix(
    test_vertices,       # [3, 3]
    trial_vertices,      # [3, 3]
    k0,
    test_shared_vertex,
    trial_shared_vertex,
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 single layer matrix for vertex-adjacent elements.

    Uses 2-region Duffy transformation for the 1/r weak singularity at the shared vertex.
    """
    points_test, points_trial, weights = get_remapped_vertex_adjacent_rule(
        order, test_shared_vertex, trial_shared_vertex
    )

    v0_test  = test_vertices[0]
    e1_test  = test_vertices[1] - v0_test
    e2_test  = test_vertices[2] - v0_test
    int_elem_test  = jnp.linalg.norm(jnp.cross(e1_test,  e2_test))  / 2.0

    v0_trial = trial_vertices[0]
    e1_trial = trial_vertices[1] - v0_trial
    e2_trial = trial_vertices[2] - v0_trial
    int_elem_trial = jnp.linalg.norm(jnp.cross(e1_trial, e2_trial)) / 2.0

    phys_test  = v0_test[None,  :] + points_test[0,  :, None] * e1_test[None,  :] + points_test[1,  :, None] * e2_test[None,  :]
    phys_trial = v0_trial[None, :] + points_trial[0, :, None] * e1_trial[None, :] + points_trial[1, :, None] * e2_trial[None, :]

    diff = phys_trial - phys_test
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))

    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = phase * M_INV_4PI / dist

    full_weights = weights * int_elem_test * int_elem_trial * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    return jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)

@partial(jit, static_argnames=['order', 'space'])
def compute_vertex_adjacent_hypersingular_matrix(
    test_vertices,       # [3, 3] - test triangle vertices
    trial_vertices,      # [3, 3] - trial triangle vertices
    test_normal,         # [3] - test element normal
    trial_normal,        # [3] - trial element normal
    k0,                  # wavenumber
    test_shared_vertex,  # local index of shared vertex in test element
    trial_shared_vertex, # local index of shared vertex in trial element
    order=4,
    space='P1',
    ):
    """
    Compute local 3x3 hypersingular matrix for vertex-adjacent elements.

    Uses 2-region Duffy transformation for vertex-adjacent singularities.
    Quadrature points are remapped based on which vertex is shared.
    """
    # Get remapped vertex-adjacent quadrature rule
    points_test, points_trial, weights = get_remapped_vertex_adjacent_rule(
        order, test_shared_vertex, trial_shared_vertex
    )

    # Test geometry
    v0_test = test_vertices[0]
    e1_test = test_vertices[1] - v0_test
    e2_test = test_vertices[2] - v0_test
    jacobian_test = jnp.stack([e1_test, e2_test], axis=1)
    cross_test = jnp.cross(e1_test, e2_test)
    int_elem_test = jnp.linalg.norm(cross_test) / 2.0

    # Trial geometry
    v0_trial = trial_vertices[0]
    e1_trial = trial_vertices[1] - v0_trial
    e2_trial = trial_vertices[2] - v0_trial
    jacobian_trial = jnp.stack([e1_trial, e2_trial], axis=1)
    cross_trial = jnp.cross(e1_trial, e2_trial)
    int_elem_trial = jnp.linalg.norm(cross_trial) / 2.0

    if space == 'P1':
        ref_gradients = jnp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        JTJ_test_inv = jnp.linalg.inv(jacobian_test.T @ jacobian_test)
        surface_curls_test = jnp.cross(
            test_normal[None, :],
            ref_gradients @ (jacobian_test @ JTJ_test_inv).T)
        JTJ_trial_inv = jnp.linalg.inv(jacobian_trial.T @ jacobian_trial)
        surface_curls_trial = jnp.cross(
            trial_normal[None, :],
            ref_gradients @ (jacobian_trial @ JTJ_trial_inv).T)
        curl_product = surface_curls_test @ surface_curls_trial.T  # [3, 3]

    # Physical points
    phys_test  = v0_test[None,  :] + points_test[0,  :, None] * e1_test[None,  :] + points_test[1,  :, None] * e2_test[None,  :]
    phys_trial = v0_trial[None, :] + points_trial[0, :, None] * e1_trial[None, :] + points_trial[1, :, None] * e2_trial[None, :]

    diff = phys_test - phys_trial
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    M_INV_4PI = 1.0 / (4.0 * jnp.pi)
    phase = jnp.exp(1j * k0 * dist)
    kernel_vals = phase * M_INV_4PI / dist

    full_weights = weights * int_elem_test * int_elem_trial * 4.0
    weighted_kernel = kernel_vals * full_weights

    basis_test  = _eval_basis(space, points_test)
    basis_trial = _eval_basis(space, points_trial)
    normal_prod = jnp.dot(test_normal, trial_normal)
    mass_term = jnp.einsum('in,jn,n->ij', basis_test, basis_trial, weighted_kernel)

    if space == 'P1':
        curl_term = jnp.sum(weighted_kernel) * curl_product
    else:
        curl_term = jnp.zeros((1, 1))

    return curl_term - k0**2 * normal_prod * mass_term