import jax
import jax.numpy as jnp

# Global dtype configuration.
# Switch here to change precision across the entire codebase.
# Note: jax_enable_x64 must be True for float64/complex128 to work.

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)
COMPLEX_DTYPE: jnp.dtype = jnp.complex64
FLOAT_DTYPE:   jnp.dtype = jnp.float32