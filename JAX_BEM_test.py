import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
import bempp_cl.api
from core.bempp_solve import bempp_solve
import time
import numpy as np
from matplotlib import pyplot as plt
from core.JAX_BEM_operators import assemble_bm
from core.JAX_BEM_fields import (
    compute_incident_field,
    compute_normal_derivative
    )
from core.JAX_BEM_kirchoff_helmholtz import propagate
from core.JAX_BEM_mesh import load_mesh, mirror_mesh
from core.sphere_analytic import sphere_analytic, mask_sphere

#%% Main solver

def bem_solve(k0, vertices, elements, normals, adjacency_data, incident_direction,
              eta, quad_order, grid_size, resolution, symmetry):
    """
    Solve BEM scattering problem using Burton-Miller formulation.

    This function is fully differentiable with respect to vertices and k0
    when adjacency_data is pre-computed.

    Args:
        k0: wavenumber
        vertices: [N, 3] vertex positions
        elements: [F, 3] triangle connectivity (GMSH format)
        normals: [F, 3] element normals
        adjacency_data: pre-computed adjacency lists from load_mesh()
        incident_direction: [3] incident wave direction
        eta: Burton-Miller coupling parameter (default: 1j)
        quad_order: quadrature order (1, 3, 4, or 7)
        grid_size: domain grid extent (cubic side length)
        resolution: grid resolution for domain solution
        symmetry: [3] boolean array for [XY, XZ, YZ] symmetry planes

    Returns:
        Tuple of (lhs, rhs, boundary_solution, domain_solution, elapsed_time)
    """

    print("Assembling operators...")
    tic_total = time.perf_counter()
    tic = time.perf_counter()

    # Single-pass Burton-Miller assembly: lhs = K - 0.5*M + eta*W
    # All three operators are accumulated into one N×N matrix (peak memory N²).
    lhs = assemble_bm(vertices, elements, normals, k0, eta, adjacency_data,
                      quad_order=quad_order, symmetry=symmetry)
    lhs.block_until_ready()

    print("Computing incident field...")
    # Compute incident field projections (L2 projection integrals)
    # These functions return projections = <phi_i, f> directly
    # Note: symmetry is NOT applied to incident field - the symmetric Green's
    # function in the operators already handles the method of images
    p_inc_proj = compute_incident_field(vertices, normals, elements,
                                        k0, incident_direction)
    dp_inc_dn_proj = compute_normal_derivative(vertices, normals, elements,
                                               k0, incident_direction)
    toc = time.perf_counter()
    print(' Operator assembly - ',round(toc-tic,3),' seconds')

    # Burton-Miller RHS: 
    rhs = -p_inc_proj + eta * dp_inc_dn_proj
    
    print("Solving with JAX GMRES...")
    tic = time.perf_counter()
    boundary_solution, info = gmres(lhs, rhs, tol=1e-5, restart=100, maxiter=1000)
    boundary_solution.block_until_ready()
    
    toc = time.perf_counter()
    toc_total = time.perf_counter()
    
    if info == 0:
        print("GMRES converged successfully")
        print('in ',round(toc-tic,3),' seconds')
    else:
        print(f"GMRES did not converge. Info: {info}")
    
    domain_solution = propagate(vertices, elements, boundary_solution,
                                k0, grid_size, resolution, symmetry=symmetry)
    domain_solution.block_until_ready()
    
    jax_time = toc_total-tic_total
    
    print("Total time: ",round(jax_time,3), "seconds")
    
    return (
            np.array(lhs), np.array(rhs),
            np.array(boundary_solution), np.array(domain_solution),
            jax_time)

#%% Main

if __name__ == "__main__":
    # Example usage
    
    # Problem parameters
    quad_order = 4 # quadrature order (1, 3, 4, or 7). 4 is bempp default.
    grid_size = 4
    resolution = 128
    eta = 1j
    c = 343.0
    freq = 250.0
    k0 = (2 * jnp.pi * freq) / c
    incident_direction = jnp.array([1.0, 0.0, 0.0])  # +x direction
    r0 = 1 # Radius of sphere for analytic solution
    # Enable/disable symmetry planes, XY, XZ, YZ
    symmetry = jnp.array([True, True, False])
    
    analytic_solution = sphere_analytic(k0, r0, incident_direction, 
                                        grid_size, resolution)
    
    # Load mesh (includes pre-computing adjacency for differentiable solver)
    mesh = bempp_cl.api.shapes.regular_sphere(int(3))
    vertices, elements, normals, adjacency_data, _ = load_mesh(mesh, symmetry)

    print(f"Wavenumber k0 = {k0:.3f}")

    # Solve
    (jax_lhs, jax_rhs,
     jax_boundary_solution, jax_domain_solution, jax_time) = bem_solve(
        k0=k0,
        vertices=vertices,
        elements=elements,
        normals=normals,
        adjacency_data=adjacency_data,
        incident_direction=incident_direction,
        eta=eta/k0,
        quad_order=quad_order,
        grid_size=grid_size,
        resolution=resolution,
        symmetry=symmetry
    )
    
    print("Solving with bempp: ")
    bp_tic = time.perf_counter()
    
    bp_boundary_solution, assembly_time = bempp_solve(k0, mesh, (1,0,0),
                                                       grid_size, resolution)
    # Propagate to domain
    bp_domain_solution = propagate(mesh.vertices.T, mesh.elements.T,
                                   jnp.array(bp_boundary_solution),
                                   k0, grid_size, resolution, symmetry=None)
    bp_domain_solution.block_until_ready()
    
    bp_toc = time.perf_counter()
    bempp_time = bp_toc - bp_tic
    
    print("bempp assembly: ",round(assembly_time,3)," seconds")
    print("bempp total: ",round(bempp_time,3)," seconds")
    
    print("Speedup: ",round(bempp_time/jax_time,3),"x")
    
    # Mask the sphere interior for accurate error calculation
    bp_domain_solution = mask_sphere(bp_domain_solution, r0, grid_size)
    jax_domain_solution = mask_sphere(jax_domain_solution, r0, grid_size)

    bp_domain_mae = np.mean(np.abs(analytic_solution - bp_domain_solution))
    jax_domain_mae = np.mean(np.abs(analytic_solution - jax_domain_solution))

    print(f"Domain MAE (bempp): {bp_domain_mae:.2e}")
    print(f"Domain MAE (JAX):   {jax_domain_mae:.2e}")

    print(f"\nJAX solution norm:   {jnp.linalg.norm(jax_domain_solution):.6f}")
    print(f"bempp solution norm: {jnp.linalg.norm(bp_domain_solution):.6f}")
    
    #%% Plots
    
    vmax = 1
    
    fig, axs = plt.subplots(3, 1, figsize=(16, 8))
    im0 = axs[0].imshow(np.real(analytic_solution[:,:,resolution//2]), cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    axs[0].set_title('Analytic solution')
    plt.colorbar(im0)
    im1 = axs[1].imshow(np.real(bp_domain_solution[:,:,resolution//2]), cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    axs[1].set_title('bempp - domain solution')
    plt.colorbar(im1)
    im2 = axs[2].imshow(np.real(jax_domain_solution[:,:,resolution//2]), cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    axs[2].set_title('JAX - domain solution')
    plt.colorbar(im2)
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(3, 1, figsize=(16, 8))
    im0 = axs[0].imshow(np.abs(analytic_solution[:,:,resolution//2]), cmap='viridis',vmin=0,vmax=vmax)
    axs[0].set_title('Analytic solution')
    plt.colorbar(im0)
    im1 = axs[1].imshow(np.abs(bp_domain_solution[:,:,resolution//2]), cmap='viridis',vmin=0,vmax=vmax)
    axs[1].set_title('bempp - domain solution')
    plt.colorbar(im1)
    im2 = axs[2].imshow(np.abs(jax_domain_solution[:,:,resolution//2]), cmap='viridis',vmin=0,vmax=vmax)
    axs[2].set_title('JAX - domain solution')
    plt.colorbar(im2)
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    im0 = axs[0].imshow(np.real(analytic_solution[:,:,resolution//2]), cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    axs[0].set_title('Analytic solution')
    plt.colorbar(im0)
    im2 = axs[1].imshow(np.real(jax_domain_solution[:,:,resolution//2]), cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    axs[1].set_title('JAX-BEM - domain solution')
    plt.colorbar(im2)
    plt.tight_layout()
    plt.show()

    #%% Show mesh using gmsh

    # Mirror across symmetry planes to get full mesh
    full_vertices, full_elements = mirror_mesh(vertices, elements, symmetry)
    grid = bempp_cl.api.Grid(full_vertices.T, full_elements.T)
    grid.plot()
