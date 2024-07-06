def make_scorer(
    score_func,
    *,
    response_method=None,
    greater_is_better=True,
    needs_proba="deprecated",
    needs_threshold="deprecated",
    **kwargs,
):
    """Make a scorer from a performance metric or loss function.

    A scorer is a wrapper around an arbitrary metric or loss function that is called
    with the signature `scorer(estimator, X, y_true, **kwargs)`.

    It is accepted in all scikit-learn estimators or functions allowing a `scoring`
    parameter.

    The parameter `response_method` allows to specify which method of the estimator
    should be used to feed the scoring/loss function.

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    response_method : {"predict_proba", "decision_function", "predict"} or \
            list/tuple of such str, default=None

        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`decision_function` or
        :term:`predict`). Possible choices are:

        - if `str`, it corresponds to the name to the method to return;
        - if a list or tuple of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.
        - if `None`, it is equivalent to `"predict"`.

        .. versionadded:: 1.4

    greater_is_better : bool, default=True
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `score_func`.

    needs_proba : bool, default=False
        Whether `score_func` requires `predict_proba` to get probability
        estimates out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

        .. deprecated:: 1.4
           `needs_proba` is deprecated in version 1.4 and will be removed in
           1.6. Use `response_method="predict_proba"` instead.

    needs_threshold : bool, default=False
        Whether `score_func` takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example `average_precision` or the area under the roc curve
        can not be computed using discrete predictions alone.

        .. deprecated:: 1.4
           `needs_threshold` is deprecated in version 1.4 and will be removed
           in 1.6. Use `response_method=("decision_function", "predict_proba")`
           instead to preserve the same behaviour.

    **kwargs : additional arguments
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, response_method='predict', beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    response_method = _get_response_method(
        response_method, needs_threshold, needs_proba
    )
    sign = 1 if greater_is_better else -1
    return _Scorer(score_func, sign, kwargs, response_method)

def get_scorer(scoring):
    """Get a scorer from string.

    Read more in the :ref:`User Guide <scoring_parameter>`.
    :func:`~sklearn.metrics.get_scorer_names` can be used to retrieve the names
    of all available scorers.

    Parameters
    ----------
    scoring : str, callable or None
        Scoring method as string. If callable it is returned as is.
        If None, returns None.

    Returns
    -------
    scorer : callable
        The scorer.

    Notes
    -----
    When passed a string, this function always returns a copy of the scorer
    object. Calling `get_scorer` twice for the same scorer results in two
    separate scorer objects.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyClassifier
    >>> from sklearn.metrics import get_scorer
    >>> X = np.reshape([0, 1, -1, -0.5, 2], (-1, 1))
    >>> y = np.array([0, 1, 1, 0, 1])
    >>> classifier = DummyClassifier(strategy="constant", constant=0).fit(X, y)
    >>> accuracy = get_scorer("accuracy")
    >>> accuracy(classifier, X, y)
    0.4
    """
    if isinstance(scoring, str):
        try:
            scorer = copy.deepcopy(_SCORERS[scoring])
        except KeyError:
            raise ValueError(
                "%r is not a valid scoring value. "
                "Use sklearn.metrics.get_scorer_names() "
                "to get valid options." % scoring
            )
    else:
        scorer = scoring
    return scorer

def check_scoring(estimator=None, scoring=None, *, allow_none=False, raise_exc=True):
    """Determine scorer from user options.

    A TypeError will be thrown if the estimator cannot be scored.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' or None, default=None
        The object to use to fit the data. If `None`, then this function may error
        depending on `allow_none`.

    scoring : str, callable, list, tuple, set, or dict, default=None
        Scorer to use. If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list, tuple or set of unique strings;
        - a callable returning a dictionary where the keys are the metric names and the
          values are the metric scorers;
        - a dictionary with metric names as keys and callables a values. The callables
          need to have the signature `callable(estimator, X, y)`.

        If None, the provided estimator object's `score` method is used.

    allow_none : bool, default=False
        Whether to return None or raise an error if no `scoring` is specified and the
        estimator has no `score` method.

    raise_exc : bool, default=True
        Whether to raise an exception (if a subset of the scorers in multimetric scoring
        fails) or to return an error code.

        - If set to `True`, raises the failing scorer's exception.
        - If set to `False`, a formatted string of the exception details is passed as
          result of the failing scorer(s).

        This applies if `scoring` is list, tuple, set, or dict. Ignored if `scoring` is
        a str or a callable.

        .. versionadded:: 1.6

    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature ``scorer(estimator, X, y)``.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.metrics import check_scoring
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> classifier = DecisionTreeClassifier(max_depth=2).fit(X, y)
    >>> scorer = check_scoring(classifier, scoring='accuracy')
    >>> scorer(classifier, X, y)
    0.96...

    >>> from sklearn.metrics import make_scorer, accuracy_score, mean_squared_log_error
    >>> X, y = load_iris(return_X_y=True)
    >>> y *= -1
    >>> clf = DecisionTreeClassifier().fit(X, y)
    >>> scoring = {
    ...     "accuracy": make_scorer(accuracy_score),
    ...     "mean_squared_log_error": make_scorer(mean_squared_log_error),
    ... }
    >>> scoring_call = check_scoring(estimator=clf, scoring=scoring, raise_exc=False)
    >>> scores = scoring_call(clf, X, y)
    >>> scores
    {'accuracy': 1.0, 'mean_squared_log_error': 'Traceback ...'}
    """
    if isinstance(scoring, str):
        return get_scorer(scoring)
    if callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("sklearn.metrics.")
            and not module.startswith("sklearn.metrics._scorer")
            and not module.startswith("sklearn.metrics.tests.")
        ):
            raise ValueError(
                "scoring value %r looks like it is a metric "
                "function rather than a scorer. A scorer should "
                "require an estimator as its first parameter. "
                "Please use `make_scorer` to convert a metric "
                "to a scorer." % scoring
            )
        return get_scorer(scoring)
    if isinstance(scoring, (list, tuple, set, dict)):
        scorers = _check_multimetric_scoring(estimator, scoring=scoring)
        return _MultimetricScorer(scorers=scorers, raise_exc=raise_exc)
    if scoring is None:
        if hasattr(estimator, "score"):
            return _PassthroughScorer(estimator)
        elif allow_none:
            return None
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not." % estimator
            )

def set_score_request(self, **kwargs):
        """Set requested parameters by the scorer.

        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Parameters
        ----------
        kwargs : dict
            Arguments should be of the form ``param_name=alias``, and `alias`
            can be one of ``{True, False, None, str}``.
        """
        if not _routing_enabled():
            raise RuntimeError(
                "This method is only available when metadata routing is enabled."
                " You can enable it using"
                " sklearn.set_config(enable_metadata_routing=True)."
            )

        self._warn_overlap(
            message=(
                "You are setting metadata request for parameters which are "
                "already set as kwargs for this metric. These set values will be "
                "overridden by passed metadata if provided. Please pass them either "
                "as metadata or kwargs to `make_scorer`."
            ),
            kwargs=kwargs,
        )
        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self

