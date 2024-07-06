def create_knn_graph_and_index(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    correct_exact_duplicates: bool = True,
    **sklearn_knn_kwargs,
) -> Tuple[csr_matrix, NearestNeighbors]:
    """Calculate the KNN graph from the features if it is not provided in the kwargs.

    Parameters
    ----------
    features :
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.
    n_neighbors :
        The number of nearest neighbors to consider. If None, a default value is determined based on the feature array size.
    metric :
        The distance metric to use for computing distances between points. If None, the metric is determined based on the feature array shape.
    correct_exact_duplicates :
        Whether to correct the KNN graph to ensure that exact duplicates have zero mutual distance, and they are correctly included in the KNN graph.
    **sklearn_knn_kwargs :
        Additional keyword arguments to be passed to the search index constructor.

    Raises
    ------
    ValueError :
        If `features` is None, as it's required to construct a KNN graph from scratch.

    Returns
    -------
    knn_graph :
        A sparse, weighted adjacency matrix representing the KNN graph of the feature array.
    knn :
        A k-nearest neighbors search object fitted to the input feature array. This object can be used to query the nearest neighbors of new data points.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index
    >>> features = np.array([
    ...    [0.701, 0.701],
    ...    [0.900, 0.436],
    ...    [0.000, 1.000],
    ... ])
    >>> knn_graph, knn = create_knn_graph_and_index(features, n_neighbors=1)
    >>> knn_graph.toarray()  # For demonstration purposes only. It is generally a bad idea to transform to dense matrix for large graphs.
    array([[0.        , 0.33140006, 0.        ],
           [0.33140006, 0.        , 0.        ],
           [0.76210367, 0.        , 0.        ]])
    >>> knn
    NearestNeighbors(metric=<function euclidean at ...>, n_neighbors=1)  # For demonstration purposes only. The actual metric may vary.
    """
    # Construct NearestNeighbors object
    knn = features_to_knn(features, n_neighbors=n_neighbors, metric=metric, **sklearn_knn_kwargs)
    # Build graph from NearestNeighbors object
    knn_graph = construct_knn_graph_from_index(knn)

    # Ensure that exact duplicates found with np.unique aren't accidentally missed in the KNN graph
    if correct_exact_duplicates:
        assert features is not None
        knn_graph = correct_knn_graph(features, knn_graph)
    return knn_graph, knn

def features_to_knn(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    **sklearn_knn_kwargs,
) -> NearestNeighbors:
    """Build and fit a k-nearest neighbors search object from an array of numerical features.

    Parameters
    ----------
    features :
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.
    n_neighbors :
        The number of nearest neighbors to consider. If None, a default value is determined based on the feature array size.
    metric :
        The distance metric to use for computing distances between points. If None, the metric is determined based on the feature array shape.
    **sklearn_knn_kwargs :
        Additional keyword arguments to be passed to the search index constructor.

    Returns
    -------
    knn :
        A k-nearest neighbors search object fitted to the input feature array.

    Examples
    --------

    >>> import numpy as np
    >>> from cleanlab.internal.neighbor import features_to_knn
    >>> features = np.random.rand(100, 10)
    >>> knn = features_to_knn(features)
    >>> knn
    NearestNeighbors(metric='cosine', n_neighbors=10)
    """
    if features is None:
        raise ValueError("Both knn and features arguments cannot be None at the same time.")
    # Use provided metric if available, otherwise decide based on the features.
    metric = metric or decide_default_metric(features)

    # Decide the number of neighbors to use in the KNN search.
    n_neighbors = _configure_num_neighbors(features, n_neighbors)

    knn = construct_knn(n_neighbors, metric, **sklearn_knn_kwargs)
    return knn.fit(features)

