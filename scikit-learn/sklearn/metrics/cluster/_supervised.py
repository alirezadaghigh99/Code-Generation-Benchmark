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

def adjusted_rand_score(labels_true, labels_pred):
    """Rand index adjusted for chance.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.

    The raw RI score is then "adjusted for chance" into the ARI score
    using the following scheme::

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation). The adjusted Rand index is bounded below by -0.5 for
    especially discordant clusterings.

    ARI is a symmetric measure::

        adjusted_rand_score(a, b) == adjusted_rand_score(b, a)

    Read more in the :ref:`User Guide <adjusted_rand_score>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=int
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=int
        Cluster labels to evaluate.

    Returns
    -------
    ARI : float
       Similarity score between -0.5 and 1.0. Random labelings have an ARI
       close to 0.0. 1.0 stands for perfect match.

    See Also
    --------
    adjusted_mutual_info_score : Adjusted Mutual Information.

    References
    ----------
    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    .. [Steinley2004] D. Steinley, Properties of the Hubert-Arabie
      adjusted Rand index, Psychological Methods 2004

    .. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

    .. [Chacon] :doi:`Minimum adjusted Rand index for two clusterings of a given size,
      2022, J. E. Chacón and A. I. Rastrojo <10.1007/s11634-022-00491-w>`

    Examples
    --------
    Perfectly matching labelings have a score of 1 even

      >>> from sklearn.metrics.cluster import adjusted_rand_score
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete but may not always be pure, hence penalized::

      >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])
      0.57...

    ARI is symmetric, so labelings that have pure clusters with members
    coming from the same classes but unnecessary splits are penalized::

      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])
      0.57...

    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::

      >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    ARI may take a negative value for especially discordant labelings that
    are a worse choice than the expected value of random labels::

      >>> adjusted_rand_score([0, 0, 1, 1], [0, 1, 0, 1])
      -0.5
    """
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

def normalized_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic"
):
    """Normalized Mutual Information between two clusterings.

    Normalized Mutual Information (NMI) is a normalization of the Mutual
    Information (MI) score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation). In this function, mutual
    information is normalized by some generalized mean of ``H(labels_true)``
    and ``H(labels_pred))``, defined by the `average_method`.

    This measure is not adjusted for chance. Therefore
    :func:`adjusted_mutual_info_score` might be preferred.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets.

    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer in the denominator.

        .. versionadded:: 0.20

        .. versionchanged:: 0.22
           The default value of ``average_method`` changed from 'geometric' to
           'arithmetic'.

    Returns
    -------
    nmi : float
       Score between 0.0 and 1.0 in normalized nats (based on the natural
       logarithm). 1.0 stands for perfectly complete labeling.

    See Also
    --------
    v_measure_score : V-Measure (NMI with arithmetic mean option).
    adjusted_rand_score : Adjusted Rand Index.
    adjusted_mutual_info_score : Adjusted Mutual Information (adjusted
        against chance).

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the NMI is null::

      >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    contingency = contingency.astype(np.float64, copy=False)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)

    # At this point mi = 0 can't be a perfect match (the special case of a single
    # cluster has been dealt with before). Hence, if mi = 0, the nmi must be 0 whatever
    # the normalization.
    if mi == 0:
        return 0.0

    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)

    normalizer = _generalized_average(h_true, h_pred, average_method)
    return mi / normalizer

def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    """Mutual Information between two clusterings.

    The Mutual Information is a measure of the similarity between two labels
    of the same data. Where :math:`|U_i|` is the number of the samples
    in cluster :math:`U_i` and :math:`|V_j|` is the number of the
    samples in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:

    .. math::

        MI(U,V)=\\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}
        \\log\\frac{N|U_i \\cap V_j|}{|U_i||V_j|}

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (i.e
    ``label_true``) with :math:`V` (i.e. ``label_pred``) will return the
    same score value. This can be useful to measure the agreement of two
    independent label assignments strategies on the same dataset when the
    real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    contingency : {array-like, sparse matrix} of shape \
            (n_classes_true, n_classes_pred), default=None
        A contingency matrix given by the
        :func:`~sklearn.metrics.cluster.contingency_matrix` function. If value
        is ``None``, it will be computed, otherwise the given value is used,
        with ``labels_true`` and ``labels_pred`` ignored.

    Returns
    -------
    mi : float
       Mutual information, a non-negative value, measured in nats using the
       natural logarithm.

    See Also
    --------
    adjusted_mutual_info_score : Adjusted against chance Mutual Information.
    normalized_mutual_info_score : Normalized Mutual Information.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    Examples
    --------
    >>> from sklearn.metrics import mutual_info_score
    >>> labels_true = [0, 1, 1, 0, 1, 0]
    >>> labels_pred = [0, 1, 0, 0, 1, 1]
    >>> mutual_info_score(labels_true, labels_pred)
    0.056...
    """
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(
            contingency,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[int, np.int32, np.int64],
        )

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    else:
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a
    # single cluster, implies MI = 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)

def fowlkes_mallows_score(labels_true, labels_pred, *, sparse=False):
    """Measure the similarity of two clusterings of a set of points.

    .. versionadded:: 0.18

    The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of
    the precision and recall::

        FMI = TP / sqrt((TP + FP) * (TP + FN))

    Where ``TP`` is the number of **True Positive** (i.e. the number of pair of
    points that belongs in the same clusters in both ``labels_true`` and
    ``labels_pred``), ``FP`` is the number of **False Positive** (i.e. the
    number of pair of points that belongs in the same clusters in
    ``labels_true`` and not in ``labels_pred``) and ``FN`` is the number of
    **False Negative** (i.e. the number of pair of points that belongs in the
    same clusters in ``labels_pred`` and not in ``labels_True``).

    The score ranges from 0 to 1. A high value indicates a good similarity
    between two clusters.

    Read more in the :ref:`User Guide <fowlkes_mallows_scores>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=int
        A clustering of the data into disjoint subsets.

    labels_pred : array-like of shape (n_samples,), dtype=int
        A clustering of the data into disjoint subsets.

    sparse : bool, default=False
        Compute contingency matrix internally with sparse matrix.

    Returns
    -------
    score : float
       The resulting Fowlkes-Mallows score.

    References
    ----------
    .. [1] `E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
       hierarchical clusterings". Journal of the American Statistical
       Association
       <https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478008>`_

    .. [2] `Wikipedia entry for the Fowlkes-Mallows Index
           <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import fowlkes_mallows_score
      >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally random, hence the FMI is null::

      >>> fowlkes_mallows_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    (n_samples,) = labels_true.shape

    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    c = c.astype(np.int64, copy=False)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    return np.sqrt(tk / pk) * np.sqrt(tk / qk) if tk != 0.0 else 0.0

