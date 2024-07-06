def sort(x: Array) -> Array:
    """Lexicographically sort the rows of a matrix, taking the columns as sequences of keys

    Args:
        x: 1D/2D Input array
    Returns:
        A sorted copy of x

    Example:
        >>> import jax.numpy as jnp
        >>> from netket.jax import sort
        >>> x = jnp.array([[1,2,3], [0,2,2], [0,1,2]])
        >>> sort(x)
        Array([[0, 1, 2],
               [0, 2, 2],
               [1, 2, 3]], dtype=int64)
    """
    if x.ndim == 1:
        return jnp.sort(x)
    else:
        return _sort_lexicographic(x)

