def nearest_neighbors(
    X,
    n_neighbors,
    metric,
    metric_kwds,
    angular,
    random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=-1,
    verbose=False,
):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    metric: string or callable
        The metric to use for the computation.

    metric_kwds: dict
        Any arguments to pass to the metric computation function.

    angular: bool
        Whether to use angular rp trees in NN approximation.

    random_state: np.random state
        The random state to use for approximate NN computations.

    low_memory: bool (optional, default True)
        Whether to pursue lower memory NNdescent.

    verbose: bool (optional, default False)
        Whether to print status data during the computation.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    rp_forest: list of trees
        The random projection forest used for searching (if used, None otherwise).
    """
    if verbose:
        print(ts(), "Finding Nearest Neighbors")

    if metric == "precomputed":
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = fast_knn_indices(X, n_neighbors)
        # knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
        # Prune any nearest neighbours that are infinite distance apart.
        disconnected_index = knn_dists == np.inf
        knn_indices[disconnected_index] = -1

        knn_search_index = None
    else:
        # TODO: Hacked values for now
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        knn_search_index = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=metric_kwds,
            random_state=random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            compressed=False,
        )
        knn_indices, knn_dists = knn_search_index.neighbor_graph

    if verbose:
        print(ts(), "Finished Nearest Neighbor Search")
    return knn_indices, knn_dists, knn_search_index

