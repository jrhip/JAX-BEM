import bempp_cl.api
from bempp_cl.api.operators.boundary import helmholtz, sparse
from bempp_cl.api.linalg import gmres
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", message="splu converted its input to CSC format")

bempp_cl.api.DEFAULT_DEVICE_INTERFACE = 'numba'

_SPACE_MAP = {
    'P1':  ('P',  1),
    'DP0': ('DP', 0),
}

def bempp_solve(k0, mesh, incident_direction, space='DP0', return_operators=False):
    """
    Solve exterior Helmholtz scattering with Burton-Miller using bempp-cl.

    BM formulation: (K - 0.5*I + eta*W) p = (-p_inc + eta * dp_inc/dn)
    with eta = i/k0.

    Args:
        k0:                wavenumber
        mesh:              bempp Grid object
        incident_direction: (3,) incident wave direction (need not be unit)
        space:             'DP0' (default) or 'P1' — matches JAX-BEM convention
        return_operators:  if True, also return the dense LHS matrix and RHS
                           projection vector for comparison with JAX-BEM

    Returns:
        coefficients:  [n_dofs] complex boundary solution
        assembly_time: float, seconds spent assembling (excludes GMRES)
        lhs_dense:     [n_dofs, n_dofs] dense weak-form LHS  (only if return_operators)
        rhs_vec:       [n_dofs] RHS projection vector         (only if return_operators)
    """
    if space not in _SPACE_MAP:
        raise ValueError(f"Unknown space {space!r}. Supported: {list(_SPACE_MAP)}")

    tic = time.perf_counter()

    eta = 1j / k0

    bempp_space = bempp_cl.api.function_space(mesh, *_SPACE_MAP[space])
    identity     = sparse.identity(bempp_space, bempp_space, bempp_space)
    double_layer = helmholtz.double_layer(bempp_space, bempp_space, bempp_space, k0)
    hypersingular = helmholtz.hypersingular(bempp_space, bempp_space, bempp_space, k0)

    burton_miller_lhs = (double_layer - 0.5 * identity) + eta * hypersingular

    direction = np.asarray(incident_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    @bempp_cl.api.complex_callable
    def p_inc_callable(x, n, domain_index, result):
        k_dot_x = k0 * (direction[0]*x[0] + direction[1]*x[1] + direction[2]*x[2])
        result[0] = np.exp(1j * k_dot_x)

    @bempp_cl.api.complex_callable
    def dp_inc_dn_callable(x, n, domain_index, result):
        k_dot_x = k0 * (direction[0]*x[0] + direction[1]*x[1] + direction[2]*x[2])
        k_dot_n = k0 * (direction[0]*n[0] + direction[1]*n[1] + direction[2]*n[2])
        result[0] = 1j * k_dot_n * np.exp(1j * k_dot_x)

    p_inc     = bempp_cl.api.GridFunction(bempp_space, fun=p_inc_callable)
    dp_inc_dn = bempp_cl.api.GridFunction(bempp_space, fun=dp_inc_dn_callable)

    burton_miller_rhs = -p_inc + eta * dp_inc_dn

    toc = time.perf_counter()
    assembly_time = toc - tic

    p_total, info = gmres(burton_miller_lhs, burton_miller_rhs, tol=1e-5)

    if return_operators:
        lhs_dense = bempp_cl.api.as_matrix(burton_miller_lhs.weak_form())
        rhs_vec   = burton_miller_rhs.projections(bempp_space)
        return p_total.coefficients, assembly_time, lhs_dense, rhs_vec

    return p_total.coefficients, assembly_time
