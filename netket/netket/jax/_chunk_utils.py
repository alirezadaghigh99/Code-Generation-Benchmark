def chunk(x, chunk_size=None):
    """
    Split an array (or a pytree of arrays) into chunks along the first axis

    Args:
        x: an array (or pytree of arrays)
        chunk_size: an integer or None (default)
            The first axis in x must be a multiple of chunk_size
    Returns: a pair (x_chunked, unchunk_fn) where
        - x_chunked is x reshaped to (-1, chunk_size)+x.shape[1:]
          if chunk_size is None then it defaults to x.shape[0], i.e. just one chunk
        - unchunk_fn is a function which restores x given x_chunked
    """
    return _chunk(x, chunk_size), _unchunkdef unchunk(x_chunked):
    """
    Merge the first two axes of an array (or a pytree of arrays)
    Args:
        x_chunked: an array (or pytree of arrays) of at least 2 dimensions
    Returns: a pair (x, chunk_fn)
        where x is x_chunked reshaped to (-1,)+x.shape[2:]
        and chunk_fn is a function which restores x given x_chunked
    """
    return _unchunk(x_chunked), partial(_chunk, chunk_size=_chunk_size(x_chunked))