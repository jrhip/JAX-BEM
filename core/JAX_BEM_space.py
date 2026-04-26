"""
Scalar BEM function space descriptors.

To add a new space, extend build_space with the new name and:
  - set n_local (DOFs per element)
  - set n_dofs  (total global DOFs)
  - set local2global [F, n_local] (local element DOF → global index)
  - add a basis evaluation branch in JAX_BEM_singular._eval_basis
"""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class FunctionSpace:
    """DOF layout for a scalar BEM function space.

    Attributes:
        name:         identifier string, e.g. 'P1' or 'DP0'
        n_local:      DOFs per element (3 for P1, 1 for DP0)
        n_dofs:       total global DOFs (N_verts for P1, N_faces for DP0)
        local2global: [F, n_local] maps local element DOF → global index
    """
    name: str
    n_local: int
    n_dofs: int
    local2global: jnp.ndarray  # [F, n_local]


def build_space(name: str, vertices, faces) -> FunctionSpace:
    """
    Create a FunctionSpace for the given mesh.

    Args:
        name:     'P1'  — continuous piecewise-linear (DOFs at vertices)
                  'DP0' — discontinuous piecewise-constant (one DOF per element)
        vertices: [N, 3] vertex positions
        faces:    [F, 3] face connectivity

    Returns:
        FunctionSpace with precomputed local2global DOF map.
    """
    n_faces = faces.shape[0]

    if name == 'P1':
        return FunctionSpace(
            name='P1',
            n_local=3,
            n_dofs=int(vertices.shape[0]),
            local2global=faces,                                         # [F, 3]
        )
    elif name == 'DP0':
        return FunctionSpace(
            name='DP0',
            n_local=1,
            n_dofs=n_faces,
            local2global=jnp.arange(n_faces, dtype=jnp.int32)[:, None],  # [F, 1]
        )
    else:
        raise ValueError(f"Unknown space {name!r}. Supported: 'P1', 'DP0'.")
