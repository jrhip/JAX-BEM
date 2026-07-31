import jax
import jax.numpy as jnp

# Global dtype configuration.
# Switch here to change precision across the entire codebase.
# Note: jax_enable_x64 must be True for float64/complex128 to work.

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)
COMPLEX_DTYPE: jnp.dtype = jnp.complex128
FLOAT_DTYPE:   jnp.dtype = jnp.float64

# Working-memory budget for one assembly tile or field-evaluation chunk, in
# megabytes.  Every block, batch and chunk size is derived from this, so peak
# memory is set here rather than by the mesh size. 
TILE_BUDGET_MB: int = 256
