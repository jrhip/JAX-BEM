# JAX-BEM
Differentiable boundary element method using JAX. 
Based on bempp-cl (https://github.com/bempp/bempp-cl)
And code by James Hipperson, Dr. Jonathan Hargreaves and Prof. Trevor Cox 
(https://acoustics.salford.ac.uk/)

# Requirements:

```pip install numpy scipy matplotlib jax bempp_cl```

Optional: gmsh for displaying 3D mesh files: https://gmsh.info/
You might need to manually set the location of the gmsh binary: 
```bempp_cl.api.GMSH_PATH = '/path/to/gmsh'```

# Usage:

Spyder recommended: https://www.spyder-ide.org/

Open JAX_BEM_test.py and run to demo the scattering from a rigid sphere benchmark, plots are displayed in-line and saved to /plots. 

JAX_BEM_errorvN.py computes the error and wall clock time with increasing mesh refinement and produces the plot from the paper. 

JAX_BEM_errorvK computes the error with increasing wavenumber k and produces the plot from the paper. 

# GPU: 

JAX largely only supports GPUs on Linux (see: https://docs.jax.dev/en/latest/installation.html)

At the top of the script set:

```jax.config.update('jax_platform_name', 'cuda')```
or
```jax.config.update('jax_platform_name', 'rocm')```

# Implemented operators

* Identity/Mass Matrix (M)
* Single Layer (V)
* Double Layer (K)
* Adjoint Double Layer (K')
* Hypersingular (W)
* Burton-Miller (combined)

The domain potential operator is the combined single layer and double layer. 
(See JAX_BEM_kirchoff_helmholtz.py)

# Implemented discretisation schemes

* P1 (continuous polynomial)
* DP0 (discontinuous polynomial)

# Citation

```
@misc{hipperson2026jaxbemgradientbasedacousticshape,
      title={JAX-BEM: Gradient-Based Acoustic Shape Optimisation via a Differentiable Boundary Element Method}, 
      author={James Hipperson and Jonathan Hargreaves and Trevor Cox},
      year={2026},
      eprint={2604.21431},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2604.21431}, 
}
```
