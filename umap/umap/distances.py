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

