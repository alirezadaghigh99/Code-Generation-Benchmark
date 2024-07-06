def chunked_parallel_special_metric(X, Y=None, metric=hellinger, chunk_size=16):
    if Y is None:
        XX, symmetrical = X, True
        row_size = col_size = X.shape[0]
    else:
        XX, symmetrical = Y, False
        row_size, col_size = X.shape[0], Y.shape[0]

    result = np.zeros((row_size, col_size), dtype=np.float32)
    n_row_chunks = (row_size // chunk_size) + 1
    for chunk_idx in numba.prange(n_row_chunks):
        n = chunk_idx * chunk_size
        chunk_end_n = min(n + chunk_size, row_size)
        m_start = n if symmetrical else 0
        for m in range(m_start, col_size, chunk_size):
            chunk_end_m = min(m + chunk_size, col_size)
            for i in range(n, chunk_end_n):
                for j in range(m, chunk_end_m):
                    result[i, j] = metric(X[i], XX[j])
    return result

def pairwise_special_metric(X, Y=None, metric="hellinger", kwds=None, force_all_finite=True):
    if callable(metric):
        if kwds is not None:
            kwd_vals = tuple(kwds.values())
        else:
            kwd_vals = ()

        @numba.njit(fastmath=True)
        def _partial_metric(_X, _Y=None):
            return metric(_X, _Y, *kwd_vals)

        return pairwise_distances(X, Y, metric=_partial_metric, force_all_finite=force_all_finite)
    else:
        special_metric_func = named_distances[metric]
    return parallel_special_metric(X, Y, metric=special_metric_func)

