def sparse_encode(
    X,
    dictionary,
    *,
    gram=None,
    cov=None,
    algorithm="lasso_lars",
    n_nonzero_coefs=None,
    alpha=None,
    copy_cov=True,
    init=None,
    max_iter=1000,
    n_jobs=None,
    check_input=True,
    verbose=0,
    positive=False,
):
    """Sparse coding.

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    dictionary : array-like of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    gram : array-like of shape (n_components, n_components), default=None
        Precomputed Gram matrix, `dictionary * dictionary'`.

    cov : array-like of shape (n_components, n_samples), default=None
        Precomputed covariance, `dictionary' * X`.

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, \
            default='lasso_lars'
        The algorithm used:

        * `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        * `'lasso_lars'`: uses Lars to compute the Lasso solution;
        * `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). lasso_lars will be faster if
          the estimated components are sparse;
        * `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        * `'threshold'`: squashes to zero all coefficients less than
          regularization from the projection `dictionary * data'`.

    n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case. If `None`, then
        `n_nonzero_coefs=int(n_features / 10)`.

    alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.
        If `None`, default to 1.

    copy_cov : bool, default=True
        Whether to copy the precomputed covariance matrix; if `False`, it may
        be overwritten.

    init : ndarray of shape (n_samples, n_components), default=None
        Initialization value of the sparse codes. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    check_input : bool, default=True
        If `False`, the input arrays X and dictionary will not be checked.

    verbose : int, default=0
        Controls the verbosity; the higher, the more messages.

    positive : bool, default=False
        Whether to enforce positivity when finding the encoding.

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse codes.

    See Also
    --------
    sklearn.linear_model.lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    sklearn.linear_model.orthogonal_mp : Solves Orthogonal Matching Pursuit problems.
    sklearn.linear_model.Lasso : Train Linear Model with L1 prior as regularizer.
    SparseCoder : Find a sparse representation of data from a fixed precomputed
        dictionary.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import sparse_encode
    >>> X = np.array([[-1, -1, -1], [0, 0, 3]])
    >>> dictionary = np.array(
    ...     [[0, 1, 0],
    ...      [-1, -1, 2],
    ...      [1, 1, 1],
    ...      [0, 1, 1],
    ...      [0, 2, 1]],
    ...    dtype=np.float64
    ... )
    >>> sparse_encode(X, dictionary, alpha=1e-10)
    array([[ 0.,  0., -1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.]])
    """
    if check_input:
        if algorithm == "lasso_cd":
            dictionary = check_array(
                dictionary, order="C", dtype=[np.float64, np.float32]
            )
            X = check_array(X, order="C", dtype=[np.float64, np.float32])
        else:
            dictionary = check_array(dictionary)
            X = check_array(X)

    if dictionary.shape[1] != X.shape[1]:
        raise ValueError(
            "Dictionary and X have different numbers of features:"
            "dictionary.shape: {} X.shape{}".format(dictionary.shape, X.shape)
        )

    _check_positive_coding(algorithm, positive)

    return _sparse_encode(
        X,
        dictionary,
        gram=gram,
        cov=cov,
        algorithm=algorithm,
        n_nonzero_coefs=n_nonzero_coefs,
        alpha=alpha,
        copy_cov=copy_cov,
        init=init,
        max_iter=max_iter,
        n_jobs=n_jobs,
        verbose=verbose,
        positive=positive,
    )

def dict_learning_online(
    X,
    n_components=2,
    *,
    alpha=1,
    max_iter=100,
    return_code=True,
    dict_init=None,
    callback=None,
    batch_size=256,
    verbose=False,
    shuffle=True,
    n_jobs=None,
    method="lars",
    random_state=None,
    positive_dict=False,
    positive_code=False,
    method_max_iter=1000,
    tol=1e-3,
    max_no_improvement=10,
):
    """Solve a dictionary learning matrix factorization problem online.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                     (U,V)
                     with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. ||.||_Fro stands for
    the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm
    which is the sum of the absolute values of all the entries in the matrix.
    This is accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    n_components : int or None, default=2
        Number of dictionary atoms to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : float, default=1
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

        .. versionadded:: 1.1

        .. deprecated:: 1.4
           `max_iter=None` is deprecated in 1.4 and will be removed in 1.6.
           Use the default value (i.e. `100`) instead.

    return_code : bool, default=True
        Whether to also return the code U or just the dictionary `V`.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the dictionary for warm restart scenarios.
        If `None`, the initial values for the dictionary are created
        with an SVD decomposition of the data via
        :func:`~sklearn.utils.extmath.randomized_svd`.

    callback : callable, default=None
        A callable that gets invoked at the end of each iteration.

    batch_size : int, default=256
        The number of samples to take in each batch.

        .. versionchanged:: 1.3
           The default value of `batch_size` changed from 3 to 256 in version 1.3.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting it in batches.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    method : {'lars', 'cd'}, default='lars'
        * `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform when solving the lasso problem.

        .. versionadded:: 0.22

    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps.

        To disable early stopping based on changes in the dictionary, set
        `tol` to 0.0.

        .. versionadded:: 1.1

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function.

        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

        .. versionadded:: 1.1

    Returns
    -------
    code : ndarray of shape (n_samples, n_components),
        The sparse code (only returned if `return_code=True`).

    dictionary : ndarray of shape (n_components, n_features),
        The solutions to the dictionary learning problem.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to `True`.

    See Also
    --------
    dict_learning : Solve a dictionary learning matrix factorization problem.
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchDictionaryLearning : A faster, less accurate, version of the dictionary
        learning algorithm.
    SparsePCA : Sparse Principal Components Analysis.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import dict_learning_online
    >>> X, _, _ = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42,
    ... )
    >>> U, V = dict_learning_online(
    ...     X, n_components=15, alpha=0.2, max_iter=20, batch_size=3, random_state=42
    ... )

    We can check the level of sparsity of `U`:

    >>> np.mean(U == 0)
    0.53...

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = U @ V
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    0.05...
    """
    # TODO(1.6): remove in 1.6
    if max_iter is None:
        warn(
            (
                "`max_iter=None` is deprecated in version 1.4 and will be removed in "
                "version 1.6. Use the default value (i.e. `100`) instead."
            ),
            FutureWarning,
        )
        max_iter = 100

    transform_algorithm = "lasso_" + method

    est = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        n_jobs=n_jobs,
        fit_algorithm=method,
        batch_size=batch_size,
        shuffle=shuffle,
        dict_init=dict_init,
        random_state=random_state,
        transform_algorithm=transform_algorithm,
        transform_alpha=alpha,
        positive_code=positive_code,
        positive_dict=positive_dict,
        transform_max_iter=method_max_iter,
        verbose=verbose,
        callback=callback,
        tol=tol,
        max_no_improvement=max_no_improvement,
    ).fit(X)

    if not return_code:
        return est.components_
    else:
        code = est.transform(X)
        return code, est.components_

def dict_learning(
    X,
    n_components,
    *,
    alpha,
    max_iter=100,
    tol=1e-8,
    method="lars",
    n_jobs=None,
    dict_init=None,
    code_init=None,
    callback=None,
    verbose=False,
    random_state=None,
    return_n_iter=False,
    positive_dict=False,
    positive_code=False,
    method_max_iter=1000,
):
    """Solve a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. ||.||_Fro stands for
    the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm
    which is the sum of the absolute values of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    n_components : int
        Number of dictionary atoms to extract.

    alpha : int or float
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}, default='lars'
        The method used:

        * `'lars'`: uses the least angle regression method to solve the lasso
           problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value for the dictionary for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    code_init : ndarray of shape (n_samples, n_components), default=None
        Initial value for the sparse code for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    callback : callable, default=None
        Callable that gets invoked every five iterations.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform.

        .. versionadded:: 0.22

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary : ndarray of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors : array
        Vector of errors at each iteration.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    See Also
    --------
    dict_learning_online : Solve a dictionary learning matrix factorization
        problem online.
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchDictionaryLearning : A faster, less accurate version
        of the dictionary learning algorithm.
    SparsePCA : Sparse Principal Components Analysis.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import dict_learning
    >>> X, _, _ = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42,
    ... )
    >>> U, V, errors = dict_learning(X, n_components=15, alpha=0.1, random_state=42)

    We can check the level of sparsity of `U`:

    >>> np.mean(U == 0)
    0.6...

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = U @ V
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    0.01...
    """
    estimator = DictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        fit_algorithm=method,
        n_jobs=n_jobs,
        dict_init=dict_init,
        callback=callback,
        code_init=code_init,
        verbose=verbose,
        random_state=random_state,
        positive_code=positive_code,
        positive_dict=positive_dict,
        transform_max_iter=method_max_iter,
    ).set_output(transform="default")
    code = estimator.fit_transform(X)
    if return_n_iter:
        return (
            code,
            estimator.components_,
            estimator.error_,
            estimator.n_iter_,
        )
    return code, estimator.components_, estimator.error_

