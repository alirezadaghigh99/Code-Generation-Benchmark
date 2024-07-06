def sigmaz(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = None
) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^z` operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    D = np.array([2 * m for m in np.arange(S, -(S + 1), -1)])
    mat = np.diag(D, 0)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)

