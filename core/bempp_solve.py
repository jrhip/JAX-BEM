import bempp_cl.api
from bempp_cl.api.operators.boundary import helmholtz, sparse
from bempp_cl.api.linalg import gmres
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", message="splu converted its input to CSC format")

bempp_cl.api.DEFAULT_DEVICE_INTERFACE = 'numba'
#bempp_cl.api.enable_console_logging()

def bempp_solve(k0, mesh, incident_direction, grid_size, resolution):
    """
    Burton-Miller formulation: combines double layer and hypersingular operators
    eta: coupling parameter (typically imaginary)
    """

    tic = time.perf_counter()
    
    eta = 1j/k0
    
    # Maths setup
    space = bempp_cl.api.function_space(mesh, "P", 1)  # Note: using P,1 not DP,0
    identity = sparse.identity(space, space, space)
    double_layer = helmholtz.double_layer(space, space, space, k0)
    hypersingular = helmholtz.hypersingular(space, space, space, k0)
    
    # Burton-Miller combined operator
    burton_miller_lhs = (double_layer - 0.5 * identity) + eta * hypersingular
    
    # Incident direction
    direction = incident_direction / np.linalg.norm(incident_direction)
    
    # Define incident field
    @bempp_cl.api.complex_callable
    def p_inc_callable(x, n, domain_index, result):
        k_dot_x = k0 * (direction[0] * x[0] + direction[1] * x[1] + direction[2] * x[2])
        result[0] = np.exp(1j * k_dot_x)
    
    # Define incident field normal derivative (rigid sphere)
    @bempp_cl.api.complex_callable
    def dp_inc_dn_callable(x, n, domain_index, result):
        k_dot_x = k0 * (direction[0] * x[0] + direction[1] * x[1] + direction[2] * x[2])
        k_dot_n = k0 * (direction[0] * n[0] + direction[1] * n[1] + direction[2] * n[2])
        result[0] = 1j * k_dot_n * np.exp(1j * k_dot_x)
        
    p_inc = bempp_cl.api.GridFunction(space, fun=p_inc_callable)
    dp_inc_dn = bempp_cl.api.GridFunction(space, fun=dp_inc_dn_callable)
    
    # Burton-Miller RHS
    burton_miller_rhs = -p_inc + eta * dp_inc_dn
    
    toc = time.perf_counter()
    assembly_time = toc - tic

    # Solve with GMRES
    p_total, info = gmres(burton_miller_lhs, burton_miller_rhs, tol=1e-5)

    return p_total.coefficients, assembly_time