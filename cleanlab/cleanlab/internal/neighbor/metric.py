def decide_default_metric(features: FeatureArray) -> Metric:
    """
    Decide the KNN metric to be used based on the shape of the feature array.

    Parameters
    ----------
    features :
        The input feature array, with shape (N, M), where N is the number of samples and M is the number of features.

    Returns
    -------
    metric :
        The distance metric to be used for neighbor search. It can be either a string
        representing the metric name ("cosine" or "euclidean") or a callable
        representing the metric function from scipy (euclidean).

    Note
    ----
    The decision of which metric to use is based on the shape of the feature array.
    If the number of columns (M) in the feature array is greater than a predefined
    cutoff value (HIGH_DIMENSION_CUTOFF), the "cosine" metric is used. This is because the cosine
    metric is more suitable for high-dimensional data.

    Otherwise, a euclidean metric is used.
    That is handled by the :py:meth:`~cleanlab.internal.neighbor.metric.decide_euclidean_metric` function.

    See Also
    --------
    HIGH_DIMENSION_CUTOFF: The cutoff value for the number of columns in the feature array.
    sklearn.metrics.pairwise.cosine_distances: The cosine metric function from scikit-learn
    """
    if features.shape[1] > HIGH_DIMENSION_CUTOFF:
        return _cosine_metric()
    return decide_euclidean_metric(features)

