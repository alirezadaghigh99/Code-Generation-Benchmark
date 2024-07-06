def pair_confusion_matrix(labels_true, labels_pred):
    """Pair confusion matrix arising from two clusterings.

    The pair confusion matrix :math:`C` computes a 2 by 2 similarity matrix
    between two clusterings by considering all pairs of samples and counting
    pairs that are assigned into the same or into different clusters under
    the true and predicted clusterings [1]_.

    Considering a pair of samples that is clustered together a positive pair,
    then as in binary classification the count of true negatives is
    :math:`C_{00}`, false negatives is :math:`C_{10}`, true positives is
    :math:`C_{11}` and false positives is :math:`C_{01}`.

    Read more in the :ref:`User Guide <pair_confusion_matrix>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.

    Returns
    -------
    C : ndarray of shape (2, 2), dtype=np.int64
        The contingency matrix.

    See Also
    --------
    sklearn.metrics.rand_score : Rand Score.
    sklearn.metrics.adjusted_rand_score : Adjusted Rand Score.
    sklearn.metrics.adjusted_mutual_info_score : Adjusted Mutual Information.

    References
    ----------
    .. [1] :doi:`Hubert, L., Arabie, P. "Comparing partitions."
           Journal of Classification 2, 193–218 (1985).
           <10.1007/BF01908075>`

    Examples
    --------
    Perfectly matching labelings have all non-zero entries on the
    diagonal regardless of actual label values:

      >>> from sklearn.metrics.cluster import pair_confusion_matrix
      >>> pair_confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
      array([[8, 0],
             [0, 4]]...

    Labelings that assign all classes members to the same clusters
    are complete but may be not always pure, hence penalized, and
    have some off-diagonal non-zero entries:

      >>> pair_confusion_matrix([0, 0, 1, 2], [0, 0, 1, 1])
      array([[8, 2],
             [0, 2]]...

    Note that the matrix is not symmetric.
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = np.int64(labels_true.shape[0])

    # Computation using the contingency data
    contingency = contingency_matrix(
        labels_true, labels_pred, sparse=True, dtype=np.int64
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data**2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares
    return C

def rand_score(labels_true, labels_pred):
    """Rand index.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings [1]_ [2]_.

    The raw RI score [3]_ is:

        RI = (number of agreeing pairs) / (number of pairs)

    Read more in the :ref:`User Guide <rand_score>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.

    Returns
    -------
    RI : float
       Similarity score between 0.0 and 1.0, inclusive, 1.0 stands for
       perfect match.

    See Also
    --------
    adjusted_rand_score: Adjusted Rand Score.
    adjusted_mutual_info_score: Adjusted Mutual Information.

    References
    ----------
    .. [1] :doi:`Hubert, L., Arabie, P. "Comparing partitions."
       Journal of Classification 2, 193–218 (1985).
       <10.1007/BF01908075>`.

    .. [2] `Wikipedia: Simple Matching Coefficient
        <https://en.wikipedia.org/wiki/Simple_matching_coefficient>`_

    .. [3] `Wikipedia: Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_

    Examples
    --------
    Perfectly matching labelings have a score of 1 even

      >>> from sklearn.metrics.cluster import rand_score
      >>> rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete but may not always be pure, hence penalized:

      >>> rand_score([0, 0, 1, 2], [0, 0, 1, 1])
      0.83...
    """
    contingency = pair_confusion_matrix(labels_true, labels_pred)
    numerator = contingency.diagonal().sum()
    denominator = contingency.sum()

    if numerator == denominator or denominator == 0:
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique
        # cluster. These are perfect matches hence return 1.0.
        return 1.0

    return numerator / denominator

def contingency_matrix(
    labels_true, labels_pred, *, eps=None, sparse=False, dtype=np.int64
):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    eps : float, default=None
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : bool, default=False
        If `True`, return a sparse CSR continency matrix. If `eps` is not
        `None` and `sparse` is `True` will raise ValueError.

        .. versionadded:: 0.18

    dtype : numeric type, default=np.int64
        Output dtype. Ignored if `eps` is not `None`.

        .. versionadded:: 0.24

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer unless set
        otherwise with the ``dtype`` argument. If ``eps`` is given, the dtype
        will be float.
        Will be a ``sklearn.sparse.csr_matrix`` if ``sparse=True``.

    Examples
    --------
    >>> from sklearn.metrics.cluster import contingency_matrix
    >>> labels_true = [0, 0, 1, 1, 2, 2]
    >>> labels_pred = [1, 0, 2, 1, 0, 2]
    >>> contingency_matrix(labels_true, labels_pred)
    array([[1, 1, 0],
           [0, 1, 1],
           [1, 0, 1]])
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix(
        (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=dtype,
    )
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency

