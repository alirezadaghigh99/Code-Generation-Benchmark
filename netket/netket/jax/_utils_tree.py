def tree_ravel(pytree: PyTree) -> tuple[jnp.ndarray, Callable]:
    """Ravel (i.e. flatten) a pytree of arrays down to a 1D array.

    Args:
      pytree: a pytree to ravel

    Returns:
      A pair where the first element is a 1D array representing the flattened and
      concatenated leaf values, and the second element is a callable for
      unflattening a 1D vector of the same length back to a pytree of of the same
      structure as the input ``pytree``.
    """
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = nkjax.vjp(_ravel_list, *leaves)
    unravel_pytree = lambda flat: tree_unflatten(treedef, unravel_list(flat))
    return flat, unravel_pytree

