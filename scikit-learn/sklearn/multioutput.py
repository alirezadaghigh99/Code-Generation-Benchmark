class MultiOutputClassifier(ClassifierMixin, _MultiOutputEstimator):
    """Multi target classification.

    This strategy consists of fitting one classifier per target. This is a
    simple strategy for extending classifiers that do not natively support
    multi-target classification.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.
        A :term:`predict_proba` method will be exposed only if `estimator` implements
        it.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    ClassifierChain : A multi-label model that arranges binary classifiers
        into a chain.
    MultiOutputRegressor : Fits one regressor per target variable.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_multilabel_classification(n_classes=3, random_state=0)
    >>> clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    >>> clf.predict(X[-2:])
    array([[1, 1, 1],
           [1, 0, 1]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, Y, sample_weight=None, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying classifier supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().fit(X, Y, sample_weight=sample_weight, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    def _check_predict_proba(self):
        if hasattr(self, "estimators_"):
            # raise an AttributeError if `predict_proba` does not exist for
            # each estimator
            [getattr(est, "predict_proba") for est in self.estimators_]
            return True
        # raise an AttributeError if `predict_proba` does not exist for the
        # unfitted estimator
        getattr(self.estimator, "predict_proba")
        return True

    @available_if(_check_predict_proba)
    def predict_proba(self, X):
        """Return prediction probabilities for each class of each output.

        This method will raise a ``ValueError`` if any of the
        estimators do not have ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        p : array of shape (n_samples, n_classes), or a list of n_outputs \
                such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

            .. versionchanged:: 0.19
                This function now returns a list of arrays where the length of
                the list is ``n_outputs``, and each array is (``n_samples``,
                ``n_classes``) for that particular output.
        """
        check_is_fitted(self)
        results = [estimator.predict_proba(X) for estimator in self.estimators_]
        return results

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        scores : float
            Mean accuracy of predicted target versus true target.
        """
        check_is_fitted(self)
        n_outputs_ = len(self.estimators_)
        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi target classification but has only one"
            )
        if y.shape[1] != n_outputs_:
            raise ValueError(
                "The number of outputs of Y for fit {0} and"
                " score {1} should be same".format(n_outputs_, y.shape[1])
            )
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))

    def _more_tags(self):
        # FIXME
        return {"_skip_test": True}

class MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator):
    """Multi target regression.

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    .. versionadded:: 0.18

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    RegressorChain : A multi-label model that arranges regressions into a
        chain.
    MultiOutputClassifier : Classifies each output independently rather than
        chaining.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_linnerud
    >>> from sklearn.multioutput import MultiOutputRegressor
    >>> from sklearn.linear_model import Ridge
    >>> X, y = load_linnerud(return_X_y=True)
    >>> regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    >>> regr.predict(X[[0]])
    array([[176..., 35..., 57...]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    @_available_if_estimator_has("partial_fit")
    def partial_fit(self, X, y, sample_weight=None, **partial_fit_params):
        """Incrementally fit the model to data, for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **partial_fit_params : dict of str -> object
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().partial_fit(X, y, sample_weight=sample_weight, **partial_fit_params)

class ClassifierChain(MetaEstimatorMixin, ClassifierMixin, _BaseChain):
    """A multi-label model that arranges binary classifiers into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    For an example of how to use ``ClassifierChain`` and benefit from its
    ensemble, see
    :ref:`ClassifierChain on a yeast dataset
    <sphx_glr_auto_examples_multioutput_plot_classifier_chain_yeast.py>` example.

    Read more in the :ref:`User Guide <classifierchain>`.

    .. versionadded:: 0.19

    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the classifier chain is built.

    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is `random` a random ordering will be used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:

        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    chain_method : {'predict', 'predict_proba', 'predict_log_proba', \
            'decision_function'} or list of such str's, default='predict'

        Prediction method to be used by estimators in the chain for
        the 'prediction' features of previous estimators in the chain.

        - if `str`, name of the method;
        - if a list of `str`, provides the method names in order of
          preference. The method used corresponds to the first method in
          the list that is implemented by `base_estimator`.

        .. versionadded:: 1.5

    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : bool, default=False
        If True, chain progress is output as each model is completed.

        .. versionadded:: 1.2

    Attributes
    ----------
    classes_ : list
        A list of arrays of length ``len(estimators_)`` containing the
        class labels for each estimator in the chain.

    estimators_ : list
        A list of clones of base_estimator.

    order_ : list
        The order of labels in the classifier chain.

    chain_method_ : str
        Prediction method used by estimators in the chain for the prediction
        features.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `base_estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    RegressorChain : Equivalent for regression.
    MultiOutputClassifier : Classifies each output independently rather than
        chaining.

    References
    ----------
    Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank, "Classifier
    Chains for Multi-label Classification", 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.multioutput import ClassifierChain
    >>> X, Y = make_multilabel_classification(
    ...    n_samples=12, n_classes=3, random_state=0
    ... )
    >>> X_train, X_test, Y_train, Y_test = train_test_split(
    ...    X, Y, random_state=0
    ... )
    >>> base_lr = LogisticRegression(solver='lbfgs', random_state=0)
    >>> chain = ClassifierChain(base_lr, order='random', random_state=0)
    >>> chain.fit(X_train, Y_train).predict(X_test)
    array([[1., 1., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    >>> chain.predict_proba(X_test)
    array([[0.8387..., 0.9431..., 0.4576...],
           [0.8878..., 0.3684..., 0.2640...],
           [0.0321..., 0.9935..., 0.0626...]])
    """

    _parameter_constraints: dict = {
        **_BaseChain._parameter_constraints,
        "chain_method": [
            list,
            tuple,
            StrOptions(
                {"predict", "predict_proba", "predict_log_proba", "decision_function"}
            ),
        ],
    }

    def __init__(
        self,
        base_estimator,
        *,
        order=None,
        cv=None,
        chain_method="predict",
        random_state=None,
        verbose=False,
    ):
        super().__init__(
            base_estimator,
            order=order,
            cv=cv,
            random_state=random_state,
            verbose=verbose,
        )
        self.chain_method = chain_method

    @_fit_context(
        # ClassifierChain.base_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of each step.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Class instance.
        """
        _raise_for_params(fit_params, self, "fit")

        super().fit(X, Y, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    @_available_if_base_estimator_has("predict_proba")
    def predict_proba(self, X):
        """Predict probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_prob : array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        return self._get_predictions(X, output_method="predict_proba")

    def predict_log_proba(self, X):
        """Predict logarithm of probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_log_prob : array-like of shape (n_samples, n_classes)
            The predicted logarithm of the probabilities.
        """
        return np.log(self.predict_proba(X))

    @_available_if_base_estimator_has("decision_function")
    def decision_function(self, X):
        """Evaluate the decision_function of the models in the chain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_decision : array-like of shape (n_samples, n_classes)
            Returns the decision function of the sample for each model
            in the chain.
        """
        return self._get_predictions(X, output_method="decision_function")

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.base_estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def _more_tags(self):
        return {"_skip_test": True, "multioutput_only": True}

