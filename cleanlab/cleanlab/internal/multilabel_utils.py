def _is_multilabel(y: np.ndarray) -> bool:
    """Checks whether `y` is in a multi-label indicator matrix format.

    Sparse matrices are not supported.
    """
    if not (isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1):
        return False
    return np.array_equal(np.unique(y), [0, 1])

