def _grid_from_X(X, percentiles, is_categorical, grid_resolution):
    """Generate a grid of points based on the percentiles of X.

    The grid is a cartesian product between the columns of ``values``. The
    ith column of ``values`` consists in ``grid_resolution`` equally-spaced
    points between the percentiles of the jth column of X.

    If ``grid_resolution`` is bigger than the number of unique values in the
    j-th column of X or if the feature is a categorical feature (by inspecting
    `is_categorical`) , then those unique values will be used instead.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_target_features)
        The data.

    percentiles : tuple of float
        The percentiles which are used to construct the extreme values of
        the grid. Must be in [0, 1].

    is_categorical : list of bool
        For each feature, tells whether it is categorical or not. If a feature
        is categorical, then the values used will be the unique ones
        (i.e. categories) instead of the percentiles.

    grid_resolution : int
        The number of equally spaced points to be placed on the grid for each
        feature.

    Returns
    -------
    grid : ndarray of shape (n_points, n_target_features)
        A value for each feature at each point in the grid. ``n_points`` is
        always ``<= grid_resolution ** X.shape[1]``.

    values : list of 1d ndarrays
        The values with which the grid has been created. The size of each
        array ``values[j]`` is either ``grid_resolution``, or the number of
        unique values in ``X[:, j]``, whichever is smaller.
    """
    if not isinstance(percentiles, Iterable) or len(percentiles) != 2:
        raise ValueError("'percentiles' must be a sequence of 2 elements.")
    if not all(0 <= x <= 1 for x in percentiles):
        raise ValueError("'percentiles' values must be in [0, 1].")
    if percentiles[0] >= percentiles[1]:
        raise ValueError("percentiles[0] must be strictly less than percentiles[1].")

    if grid_resolution <= 1:
        raise ValueError("'grid_resolution' must be strictly greater than 1.")

    values = []
    # TODO: we should handle missing values (i.e. `np.nan`) specifically and store them
    # in a different Bunch attribute.
    for feature, is_cat in enumerate(is_categorical):
        try:
            uniques = np.unique(_safe_indexing(X, feature, axis=1))
        except TypeError as exc:
            # `np.unique` will fail in the presence of `np.nan` and `str` categories
            # due to sorting. Temporary, we reraise an error explaining the problem.
            raise ValueError(
                f"The column #{feature} contains mixed data types. Finding unique "
                "categories fail due to sorting. It usually means that the column "
                "contains `np.nan` values together with `str` categories. Such use "
                "case is not yet supported in scikit-learn."
            ) from exc
        if is_cat or uniques.shape[0] < grid_resolution:
            # Use the unique values either because:
            # - feature has low resolution use unique values
            # - feature is categorical
            axis = uniques
        else:
            # create axis based on percentiles and grid resolution
            emp_percentiles = mquantiles(
                _safe_indexing(X, feature, axis=1), prob=percentiles, axis=0
            )
            if np.allclose(emp_percentiles[0], emp_percentiles[1]):
                raise ValueError(
                    "percentiles are too close to each other, "
                    "unable to build the grid. Please choose percentiles "
                    "that are further apart."
                )
            axis = np.linspace(
                emp_percentiles[0],
                emp_percentiles[1],
                num=grid_resolution,
                endpoint=True,
            )
        values.append(axis)

    return cartesian(values), values

def partial_dependence(
    estimator,
    X,
    features,
    *,
    sample_weight=None,
    categorical_features=None,
    feature_names=None,
    response_method="auto",
    percentiles=(0.05, 0.95),
    grid_resolution=100,
    method="auto",
    kind="average",
):
    """Partial dependence of ``features``.

    Partial dependence of a feature (or a set of features) corresponds to
    the average response of an estimator for each possible value of the
    feature.

    Read more in the :ref:`User Guide <partial_dependence>`.

    .. warning::

        For :class:`~sklearn.ensemble.GradientBoostingClassifier` and
        :class:`~sklearn.ensemble.GradientBoostingRegressor`, the
        `'recursion'` method (used by default) will not account for the `init`
        predictor of the boosting process. In practice, this will produce
        the same values as `'brute'` up to a constant offset in the target
        response, provided that `init` is a constant estimator (which is the
        default). However, if `init` is not a constant estimator, the
        partial dependence values are incorrect for `'recursion'` because the
        offset will be sample-dependent. It is preferable to use the `'brute'`
        method. Note that this only applies to
        :class:`~sklearn.ensemble.GradientBoostingClassifier` and
        :class:`~sklearn.ensemble.GradientBoostingRegressor`, not to
        :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing :term:`predict`,
        :term:`predict_proba`, or :term:`decision_function`.
        Multioutput-multiclass classifiers are not supported.

    X : {array-like, sparse matrix or dataframe} of shape (n_samples, n_features)
        ``X`` is used to generate a grid of values for the target
        ``features`` (where the partial dependence will be evaluated), and
        also to generate values for the complement features when the
        `method` is 'brute'.

    features : array-like of {int, str, bool} or int or str
        The feature (e.g. `[0]`) or pair of interacting features
        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights are used to calculate weighted means when averaging the
        model output. If `None`, then samples are equally weighted. If
        `sample_weight` is not `None`, then `method` will be set to `'brute'`.
        Note that `sample_weight` is ignored for `kind='individual'`.

        .. versionadded:: 1.3

    categorical_features : array-like of shape (n_features,) or shape \
            (n_categorical_features,), dtype={bool, int, str}, default=None
        Indicates the categorical features.

        - `None`: no feature will be considered categorical;
        - boolean array-like: boolean mask of shape `(n_features,)`
            indicating which features are categorical. Thus, this array has
            the same shape has `X.shape[1]`;
        - integer or string array-like: integer indices or strings
            indicating categorical features.

        .. versionadded:: 1.2

    feature_names : array-like of shape (n_features,), dtype=str, default=None
        Name of each feature; `feature_names[i]` holds the name of the feature
        with index `i`.
        By default, the name of the feature corresponds to their numerical
        index for NumPy array and their column name for pandas dataframe.

        .. versionadded:: 1.2

    response_method : {'auto', 'predict_proba', 'decision_function'}, \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist. If
        ``method`` is 'recursion', the response is always the output of
        :term:`decision_function`.

    percentiles : tuple of float, default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme values
        for the grid. Must be in [0, 1].

    grid_resolution : int, default=100
        The number of equally spaced points on the grid, for each target
        feature.

    method : {'auto', 'recursion', 'brute'}, default='auto'
        The method used to calculate the averaged predictions:

        - `'recursion'` is only supported for some tree-based estimators
          (namely
          :class:`~sklearn.ensemble.GradientBoostingClassifier`,
          :class:`~sklearn.ensemble.GradientBoostingRegressor`,
          :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,
          :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,
          :class:`~sklearn.tree.DecisionTreeRegressor`,
          :class:`~sklearn.ensemble.RandomForestRegressor`,
          ) when `kind='average'`.
          This is more efficient in terms of speed.
          With this method, the target response of a
          classifier is always the decision function, not the predicted
          probabilities. Since the `'recursion'` method implicitly computes
          the average of the Individual Conditional Expectation (ICE) by
          design, it is not compatible with ICE and thus `kind` must be
          `'average'`.

        - `'brute'` is supported for any estimator, but is more
          computationally intensive.

        - `'auto'`: the `'recursion'` is used for estimators that support it,
          and `'brute'` is used otherwise. If `sample_weight` is not `None`,
          then `'brute'` is used regardless of the estimator.

        Please see :ref:`this note <pdp_method_differences>` for
        differences between the `'brute'` and `'recursion'` method.

    kind : {'average', 'individual', 'both'}, default='average'
        Whether to return the partial dependence averaged across all the
        samples in the dataset or one value per sample or both.
        See Returns below.

        Note that the fast `method='recursion'` option is only available for
        `kind='average'` and `sample_weights=None`. Computing individual
        dependencies and doing weighted averages requires using the slower
        `method='brute'`.

        .. versionadded:: 0.24

    Returns
    -------
    predictions : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        individual : ndarray of shape (n_outputs, n_instances, \
                len(values[0]), len(values[1]), ...)
            The predictions for all the points in the grid for all
            samples in X. This is also known as Individual
            Conditional Expectation (ICE).
            Only available when `kind='individual'` or `kind='both'`.

        average : ndarray of shape (n_outputs, len(values[0]), \
                len(values[1]), ...)
            The predictions for all the points in the grid, averaged
            over all samples in X (or over the training data if
            `method` is 'recursion').
            Only available when `kind='average'` or `kind='both'`.

        grid_values : seq of 1d ndarrays
            The values with which the grid has been created. The generated
            grid is a cartesian product of the arrays in `grid_values` where
            `len(grid_values) == len(features)`. The size of each array
            `grid_values[j]` is either `grid_resolution`, or the number of
            unique values in `X[:, j]`, whichever is smaller.

            .. versionadded:: 1.3

        `n_outputs` corresponds to the number of classes in a multi-class
        setting, or to the number of tasks for multi-output regression.
        For classical regression and binary classification `n_outputs==1`.
        `n_values_feature_j` corresponds to the size `grid_values[j]`.

    See Also
    --------
    PartialDependenceDisplay.from_estimator : Plot Partial Dependence.
    PartialDependenceDisplay : Partial Dependence visualization.

    Examples
    --------
    >>> X = [[0, 0, 2], [1, 0, 0]]
    >>> y = [0, 1]
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)
    >>> partial_dependence(gb, features=[0], X=X, percentiles=(0, 1),
    ...                    grid_resolution=2) # doctest: +SKIP
    (array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])
    """
    check_is_fitted(estimator)

    if not (is_classifier(estimator) or is_regressor(estimator)):
        raise ValueError("'estimator' must be a fitted regressor or classifier.")

    if is_classifier(estimator) and isinstance(estimator.classes_[0], np.ndarray):
        raise ValueError("Multiclass-multioutput estimators are not supported")

    # Use check_array only on lists and other non-array-likes / sparse. Do not
    # convert DataFrame into a NumPy array.
    if not (hasattr(X, "__array__") or sparse.issparse(X)):
        X = check_array(X, force_all_finite="allow-nan", dtype=object)

    if is_regressor(estimator) and response_method != "auto":
        raise ValueError(
            "The response_method parameter is ignored for regressors and "
            "must be 'auto'."
        )

    if kind != "average":
        if method == "recursion":
            raise ValueError(
                "The 'recursion' method only applies when 'kind' is set to 'average'"
            )
        method = "brute"

    if method == "recursion" and sample_weight is not None:
        raise ValueError(
            "The 'recursion' method can only be applied when sample_weight is None."
        )

    if method == "auto":
        if sample_weight is not None:
            method = "brute"
        elif isinstance(estimator, BaseGradientBoosting) and estimator.init is None:
            method = "recursion"
        elif isinstance(
            estimator,
            (BaseHistGradientBoosting, DecisionTreeRegressor, RandomForestRegressor),
        ):
            method = "recursion"
        else:
            method = "brute"

    if method == "recursion":
        if not isinstance(
            estimator,
            (
                BaseGradientBoosting,
                BaseHistGradientBoosting,
                DecisionTreeRegressor,
                RandomForestRegressor,
            ),
        ):
            supported_classes_recursion = (
                "GradientBoostingClassifier",
                "GradientBoostingRegressor",
                "HistGradientBoostingClassifier",
                "HistGradientBoostingRegressor",
                "HistGradientBoostingRegressor",
                "DecisionTreeRegressor",
                "RandomForestRegressor",
            )
            raise ValueError(
                "Only the following estimators support the 'recursion' "
                "method: {}. Try using method='brute'.".format(
                    ", ".join(supported_classes_recursion)
                )
            )
        if response_method == "auto":
            response_method = "decision_function"

        if response_method != "decision_function":
            raise ValueError(
                "With the 'recursion' method, the response_method must be "
                "'decision_function'. Got {}.".format(response_method)
            )

    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    if _determine_key_type(features, accept_slice=False) == "int":
        # _get_column_indices() supports negative indexing. Here, we limit
        # the indexing to be positive. The upper bound will be checked
        # by _get_column_indices()
        if np.any(np.less(features, 0)):
            raise ValueError("all features must be in [0, {}]".format(X.shape[1] - 1))

    features_indices = np.asarray(
        _get_column_indices(X, features), dtype=np.intp, order="C"
    ).ravel()

    feature_names = _check_feature_names(X, feature_names)

    n_features = X.shape[1]
    if categorical_features is None:
        is_categorical = [False] * len(features_indices)
    else:
        categorical_features = np.asarray(categorical_features)
        if categorical_features.dtype.kind == "b":
            # categorical features provided as a list of boolean
            if categorical_features.size != n_features:
                raise ValueError(
                    "When `categorical_features` is a boolean array-like, "
                    "the array should be of shape (n_features,). Got "
                    f"{categorical_features.size} elements while `X` contains "
                    f"{n_features} features."
                )
            is_categorical = [categorical_features[idx] for idx in features_indices]
        elif categorical_features.dtype.kind in ("i", "O", "U"):
            # categorical features provided as a list of indices or feature names
            categorical_features_idx = [
                _get_feature_index(cat, feature_names=feature_names)
                for cat in categorical_features
            ]
            is_categorical = [
                idx in categorical_features_idx for idx in features_indices
            ]
        else:
            raise ValueError(
                "Expected `categorical_features` to be an array-like of boolean,"
                f" integer, or string. Got {categorical_features.dtype} instead."
            )

    grid, values = _grid_from_X(
        _safe_indexing(X, features_indices, axis=1),
        percentiles,
        is_categorical,
        grid_resolution,
    )

    if method == "brute":
        averaged_predictions, predictions = _partial_dependence_brute(
            estimator, grid, features_indices, X, response_method, sample_weight
        )

        # reshape predictions to
        # (n_outputs, n_instances, n_values_feature_0, n_values_feature_1, ...)
        predictions = predictions.reshape(
            -1, X.shape[0], *[val.shape[0] for val in values]
        )
    else:
        averaged_predictions = _partial_dependence_recursion(
            estimator, grid, features_indices
        )

    # reshape averaged_predictions to
    # (n_outputs, n_values_feature_0, n_values_feature_1, ...)
    averaged_predictions = averaged_predictions.reshape(
        -1, *[val.shape[0] for val in values]
    )
    pdp_results = Bunch(grid_values=values)

    if kind == "average":
        pdp_results["average"] = averaged_predictions
    elif kind == "individual":
        pdp_results["individual"] = predictions
    else:  # kind='both'
        pdp_results["average"] = averaged_predictions
        pdp_results["individual"] = predictions

    return pdp_results

