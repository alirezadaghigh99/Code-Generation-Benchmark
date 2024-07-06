def tswspectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    method=None,
    tol=0.0,
    maxiter=0
):
    """Given a graph, compute the spectral embedding of the graph. This is
    simply the eigenvectors of the Laplacian of the graph. Here we use the
    normalized laplacian and a truncated SVD-based guess of the
    eigenvectors to "warm" up the eigensolver. This function should
    give results of similar accuracy to the spectral_layout function, but
    may converge more quickly for graph Laplacians that cause
    spectral_layout to take an excessive amount of time to complete.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data

    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
        Used only if the multiple connected components are found in the
        graph.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'.
        Used only if the multiple connected components are found in the
        graph.

    method: str (optional, default None, values either 'eigsh' or 'lobpcg')
        Name of the eigenvalue computation method to use to compute the spectral
        embedding. If left to None (or empty string), as by default, the method is
        chosen from the number of vectors in play: larger vector collections are
        handled with lobpcg, smaller collections with eigsh. Method names correspond
        to SciPy routines in scipy.sparse.linalg.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    return _spectral_layout(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        init="tsvd",
        method=method,
        tol=tol,
        maxiter=maxiter
    )

