def _materialize_array(matvec, shape, dtype=None):
  """Materializes the matrix A used in matvec(x) = Ax."""
  x = jnp.zeros(shape, dtype)
  return jax.jacfwd(matvec)(x)