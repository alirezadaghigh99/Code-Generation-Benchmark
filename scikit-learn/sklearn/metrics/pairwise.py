def pairwise_distances(
    X,
    Y=None,
    metric="euclidean",
    *,
    n_jobs=None,
    force_all_finite=True,
    **kwds,
):
    """Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns
    a distance matrix.
    If the input is a vector array, the distances are computed.
    If the input is a distances matrix, it is returned instead.
    If the input is a collection of non-numeric data (e.g. a list of strings or a
    boolean array), a custom metric must be passed.

    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix
      inputs.
      ['nan_euclidean'] but it does not yet support sparse matrices.

    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    .. note::
        `'kulsinski'` is deprecated from SciPy 1.9 and will be removed in SciPy 1.11.

    .. note::
        `'matching'` has been removed in SciPy 1.9 (use `'hamming'` instead).

    Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
    valid scipy.spatial.distance metrics), the scikit-learn implementation
    will be used, which is faster and has support for sparse matrices (except
    for 'cityblock'). For a verbose description of the metrics from
    scikit-learn, see :func:`sklearn.metrics.pairwise.distance_metrics`
    function.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise distances between samples, or a feature array.
        The shape of the array should be (n_samples_X, n_samples_X) if
        metric == "precomputed" and (n_samples_X, n_features) otherwise.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. Only allowed if
        metric != "precomputed".

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in ``pairwise.PAIRWISE_DISTANCE_FUNCTIONS``.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        The "euclidean" and "cosine" metrics rely heavily on BLAS which is already
        multithreaded. So, increasing `n_jobs` would likely cause oversubscription
        and quickly degrade performance.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. Ignored
        for a metric listed in ``pairwise.PAIRWISE_DISTANCE_FUNCTIONS``. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.22
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    D : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_samples_Y)
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.

    See Also
    --------
    pairwise_distances_chunked : Performs the same calculation as this
        function, but returns a generator of chunks of the distance matrix, in
        order to limit memory usage.
    sklearn.metrics.pairwise.paired_distances : Computes the distances between
        corresponding elements of two arrays.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import pairwise_distances
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> pairwise_distances(X, Y, metric='sqeuclidean')
    array([[1., 2.],
           [2., 1.]])
    """
    if metric == "precomputed":
        X, _ = check_pairwise_arrays(
            X, Y, precomputed=True, force_all_finite=force_all_finite
        )

        whom = (
            "`pairwise_distances`. Precomputed distance "
            " need to have non-negative values."
        )
        check_non_negative(X, whom=whom)
        return X
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(
            _pairwise_callable,
            metric=metric,
            force_all_finite=force_all_finite,
            **kwds,
        )
    else:
        if issparse(X) or issparse(Y):
            raise TypeError("scipy distance metrics do not support sparse matrices.")

        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else "infer_float"

        if dtype == bool and (X.dtype != bool or (Y is not None and Y.dtype != bool)):
            msg = "Data was converted to boolean for metric %s" % metric
            warnings.warn(msg, DataConversionWarning)

        X, Y = check_pairwise_arrays(
            X, Y, dtype=dtype, force_all_finite=force_all_finite
        )

        # precompute data-derived metric params
        params = _precompute_metric_params(X, Y, metric=metric, **kwds)
        kwds.update(**params)

        if effective_n_jobs(n_jobs) == 1 and X is Y:
            return distance.squareform(distance.pdist(X, metric=metric, **kwds))
        func = partial(distance.cdist, metric=metric, **kwds)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)

def check_pairwise_arrays(
    X,
    Y,
    *,
    precomputed=False,
    dtype="infer_float",
    accept_sparse="csr",
    force_all_finite=True,
    ensure_2d=True,
    copy=False,
):
    """Set X and Y appropriately and checks inputs.

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)

    precomputed : bool, default=False
        True if X is to be treated as precomputed distances to the samples in
        Y.

    dtype : str, type, list of type or None default="infer_float"
        Data type required for X and Y. If "infer_float", the dtype will be an
        appropriate float type selected by _return_float_dtype. If None, the
        dtype of the input is preserved.

        .. versionadded:: 0.18

    accept_sparse : str, bool or list/tuple of str, default='csr'
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.22
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`.

    ensure_2d : bool, default=True
        Whether to raise an error when the input arrays are not 2-dimensional. Setting
        this to `False` is necessary when using a custom metric with certain
        non-numerical inputs (e.g. a list of strings).

        .. versionadded:: 1.5

    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

        .. versionadded:: 0.22

    Returns
    -------
    safe_X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    xp, _ = get_namespace(X, Y)
    if any([issparse(X), issparse(Y)]) or _is_numpy_namespace(xp):
        X, Y, dtype_float = _return_float_dtype(X, Y)
    else:
        dtype_float = _find_matching_floating_dtype(X, Y, xp=xp)

    estimator = "check_pairwise_arrays"
    if dtype == "infer_float":
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(
            X,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
            ensure_2d=ensure_2d,
        )
    else:
        X = check_array(
            X,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
            ensure_2d=ensure_2d,
        )
        Y = check_array(
            Y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
            ensure_2d=ensure_2d,
        )

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(
                "Precomputed metric requires shape "
                "(n_queries, n_indexed). Got (%d, %d) "
                "for %d indexed." % (X.shape[0], X.shape[1], Y.shape[0])
            )
    elif ensure_2d and X.shape[1] != Y.shape[1]:
        # Only check the number of features if 2d arrays are enforced. Otherwise,
        # validation is left to the user for custom metrics.
        raise ValueError(
            "Incompatible dimension for X and Y matrices: "
            "X.shape[1] == %d while Y.shape[1] == %d" % (X.shape[1], Y.shape[1])
        )

    return X, Y

def pairwise_distances_chunked(
    X,
    Y=None,
    *,
    reduce_func=None,
    metric="euclidean",
    n_jobs=None,
    working_memory=None,
    **kwds,
):
    """Generate a distance matrix chunk by chunk with optional reduction.

    In cases where not all of a pairwise distance matrix needs to be
    stored at once, this is used to calculate pairwise distances in
    ``working_memory``-sized chunks.  If ``reduce_func`` is given, it is
    run on each chunk and its return values are concatenated into lists,
    arrays or sparse matrices.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise distances between samples, or a feature array.
        The shape the array should be (n_samples_X, n_samples_X) if
        metric='precomputed' and (n_samples_X, n_features) otherwise.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. Only allowed if
        metric != "precomputed".

    reduce_func : callable, default=None
        The function which is applied on each chunk of the distance matrix,
        reducing it to needed values.  ``reduce_func(D_chunk, start)``
        is called repeatedly, where ``D_chunk`` is a contiguous vertical
        slice of the pairwise distance matrix, starting at row ``start``.
        It should return one of: None; an array, a list, or a sparse matrix
        of length ``D_chunk.shape[0]``; or a tuple of such objects.
        Returning None is useful for in-place operations, rather than
        reductions.

        If None, pairwise_distances_chunked returns a generator of vertical
        chunks of the distance matrix.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter,
        or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on
        each pair of instances (rows) and the resulting value recorded.
        The callable should take two arrays from X as input and return a
        value indicating the distance between them.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by
        breaking down the pairwise matrix into n_jobs even slices and
        computing them in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    working_memory : float, default=None
        The sought maximum memory for temporary distance matrix chunks.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Yields
    ------
    D_chunk : {ndarray, sparse matrix}
        A contiguous slice of distance matrix, optionally processed by
        ``reduce_func``.

    Examples
    --------
    Without reduce_func:

    >>> import numpy as np
    >>> from sklearn.metrics import pairwise_distances_chunked
    >>> X = np.random.RandomState(0).rand(5, 3)
    >>> D_chunk = next(pairwise_distances_chunked(X))
    >>> D_chunk
    array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
           [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
           [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
           [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
           [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])

    Retrieve all neighbors and average distance within radius r:

    >>> r = .2
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
    ...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
    ...     return neigh, avg_dist
    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
    >>> neigh, avg_dist = next(gen)
    >>> neigh
    [array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
    >>> avg_dist
    array([0.039..., 0.        , 0.        , 0.039..., 0.        ])

    Where r is defined per sample, we need to make use of ``start``:

    >>> r = [.2, .4, .4, .3, .1]
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r[i])
    ...              for i, d in enumerate(D_chunk, start)]
    ...     return neigh
    >>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
    >>> neigh
    [array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]

    Force row-by-row generation by reducing ``working_memory``:

    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
    ...                                  working_memory=0)
    >>> next(gen)
    [array([0, 3])]
    >>> next(gen)
    [array([0, 1])]
    """
    n_samples_X = _num_samples(X)
    if metric == "precomputed":
        slices = (slice(0, n_samples_X),)
    else:
        if Y is None:
            Y = X
        # We get as many rows as possible within our working_memory budget to
        # store len(Y) distances in each row of output.
        #
        # Note:
        #  - this will get at least 1 row, even if 1 row of distances will
        #    exceed working_memory.
        #  - this does not account for any temporary memory usage while
        #    calculating distances (e.g. difference of vectors in manhattan
        #    distance.
        chunk_n_rows = get_chunk_n_rows(
            row_bytes=8 * _num_samples(Y),
            max_n_rows=n_samples_X,
            working_memory=working_memory,
        )
        slices = gen_batches(n_samples_X, chunk_n_rows)

    # precompute data-derived metric params
    params = _precompute_metric_params(X, Y, metric=metric, **kwds)
    kwds.update(**params)

    for sl in slices:
        if sl.start == 0 and sl.stop == n_samples_X:
            X_chunk = X  # enable optimised paths for X is Y
        else:
            X_chunk = X[sl]
        D_chunk = pairwise_distances(X_chunk, Y, metric=metric, n_jobs=n_jobs, **kwds)
        if (X is Y or Y is None) and PAIRWISE_DISTANCE_FUNCTIONS.get(
            metric, None
        ) is euclidean_distances:
            # zeroing diagonal, taking care of aliases of "euclidean",
            # i.e. "l2"
            D_chunk.flat[sl.start :: _num_samples(X) + 1] = 0
        if reduce_func is not None:
            chunk_size = D_chunk.shape[0]
            D_chunk = reduce_func(D_chunk, sl.start)
            _check_chunk_size(D_chunk, chunk_size)
        yield D_chunk

def nan_euclidean_distances(
    X, Y=None, *, squared=False, missing_values=np.nan, copy=True
):
    """Calculate the euclidean distances in the presence of missing values.

    Compute the euclidean distance between each pair of samples in X and Y,
    where Y=X is assumed if Y=None. When calculating the distance between a
    pair of samples, this formulation ignores feature coordinates with a
    missing value in either sample and scales up the weight of the remaining
    coordinates:

        dist(x,y) = sqrt(weight * sq. distance from present coordinates)
        where,
        weight = Total # of coordinates / # of present coordinates

    For example, the distance between ``[3, na, na, 6]`` and ``[1, na, 4, 5]``
    is:

        .. math::
            \\sqrt{\\frac{4}{2}((3-1)^2 + (6-5)^2)}

    If all the coordinates are missing or if there are no common present
    coordinates then NaN is returned for that pair.

    Read more in the :ref:`User Guide <metrics>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : array-like of shape (n_samples_Y, n_features), default=None
        An array where each row is a sample and each column is a feature.
        If `None`, method uses `Y=X`.

    squared : bool, default=False
        Return squared Euclidean distances.

    missing_values : np.nan, float or int, default=np.nan
        Representation of missing value.

    copy : bool, default=True
        Make and use a deep copy of X and Y (if Y exists).

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.

    See Also
    --------
    paired_distances : Distances between pairs of elements of X and Y.

    References
    ----------
    * John K. Dixon, "Pattern Recognition with Partly Missing Data",
      IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
      10, pp. 617 - 621, Oct. 1979.
      http://ieeexplore.ieee.org/abstract/document/4310090/

    Examples
    --------
    >>> from sklearn.metrics.pairwise import nan_euclidean_distances
    >>> nan = float("NaN")
    >>> X = [[0, 1], [1, nan]]
    >>> nan_euclidean_distances(X, X) # distance between rows of X
    array([[0.        , 1.41421356],
           [1.41421356, 0.        ]])

    >>> # get distance to origin
    >>> nan_euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    """

    force_all_finite = "allow-nan" if is_scalar_nan(missing_values) else True
    X, Y = check_pairwise_arrays(
        X, Y, accept_sparse=False, force_all_finite=force_all_finite, copy=copy
    )
    # Get missing mask for X
    missing_X = _get_mask(X, missing_values)

    # Get missing mask for Y
    missing_Y = missing_X if Y is X else _get_mask(Y, missing_values)

    # set missing values to zero
    X[missing_X] = 0
    Y[missing_Y] = 0

    distances = euclidean_distances(X, Y, squared=True)

    # Adjust distances for missing values
    XX = X * X
    YY = Y * Y
    distances -= np.dot(XX, missing_Y.T)
    distances -= np.dot(missing_X, YY.T)

    np.clip(distances, 0, None, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        np.fill_diagonal(distances, 0.0)

    present_X = 1 - missing_X
    present_Y = present_X if Y is X else ~missing_Y
    present_count = np.dot(present_X, present_Y.T)
    distances[present_count == 0] = np.nan
    # avoid divide by zero
    np.maximum(1, present_count, out=present_count)
    distances /= present_count
    distances *= X.shape[1]

    if not squared:
        np.sqrt(distances, out=distances)

    return distances

