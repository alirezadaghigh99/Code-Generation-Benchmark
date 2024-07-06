def data_shapley_knn(
    labels: np.ndarray,
    *,
    features: Optional[np.ndarray] = None,
    knn_graph: Optional[csr_matrix] = None,
    metric: Optional[Union[str, Callable]] = None,
    k: int = 10,
) -> np.ndarray:
    """
    Compute the Data Shapley values of data points using a K-Nearest Neighbors (KNN) graph.

    This function calculates the contribution (Data Shapley value) of each data point in a dataset
    for model training, either directly from data features or using a precomputed KNN graph.

    The examples in the dataset with lowest data valuation scores contribute least
    to a trained ML model’s performance (those whose value falls below a threshold are flagged with this type of issue).
    The data valuation score is an approximate Data Shapley value, calculated based on the labels of the top k nearest neighbors of an example. Details on this KNN-Shapley value can be found in these papers:
    https://arxiv.org/abs/1908.08619 and https://arxiv.org/abs/1911.07128.

    Parameters
    ----------
    labels :
        An array of labels for the data points(only for multi-class classification datasets).
    features :
        Feature embeddings (vector representations) of every example in the dataset.

            Necessary if `knn_graph` is not supplied.

            If provided, this must be a 2D array with shape (num_examples, num_features).
    knn_graph :
        A precomputed sparse KNN graph. If not provided, it will be computed from the `features` using the specified `metric`.
    metric : Optional[str or Callable], default=None
        The distance metric for KNN graph construction.
        Supports metrics available in ``sklearn.neighbors.NearestNeighbors``
        Default metric is ``"cosine"`` for ``dim(features) > 3``, otherwise ``"euclidean"`` for lower-dimensional data.
        The euclidean is computed with an efficient implementation from scikit-learn when the number of examples is greater than 100.
        When the number of examples is 100 or fewer, a more numerically stable version of the euclidean distance from scipy is used.
    k :
        The number of neighbors to consider for the KNN graph and Data Shapley value computation.
        Must be less than the total number of data points.
        The value may not exceed the number of neighbors of each data point stored in the KNN graph.

    Returns
    -------
    scores :
        An array of transformed Data Shapley values for each data point, calibrated to indicate their relative importance.
        These scores have been adjusted to fall within 0 to 1.
        Values closer to 1 indicate data points that are highly influential and positively contribute to a trained ML model's performance.
        Conversely, scores below 0.5 indicate data points estimated to negatively impact model performance.

    Raises
    ------
    ValueError
        If neither `knn_graph` nor `features` are provided, or if `k` is larger than the number of examples in `features`.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.data_valuation import data_shapley_knn
    >>> labels = np.array([0, 1, 0, 1, 0])
    >>> features = np.array([[0, 1, 2, 3, 4]]).T
    >>> data_shapley_knn(labels=labels, features=features, k=4)
    array([0.55 , 0.525, 0.55 , 0.525, 0.55 ])
    """
    if knn_graph is None and features is None:
        raise ValueError("Either knn_graph or features must be provided.")

    # Use provided knn_graph or compute it from features
    if knn_graph is None:
        knn_graph, _ = create_knn_graph_and_index(features, n_neighbors=k, metric=metric)

    num_examples = labels.shape[0]
    distances = knn_graph.indices.reshape(num_examples, -1)
    scores = _knn_shapley_score(neighbor_indices=distances, y=labels, k=k)
    return 0.5 * (scores + 1)

