def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        n_bins=5,
        strategy="uniform",
        pos_label=None,
        name=None,
        ref_line=True,
        ax=None,
        **kwargs,
    ):
        """Plot calibration curve using a binary classifier and data.

        A calibration curve, also known as a reliability diagram, uses inputs
        from a binary classifier and plots the average predicted probability
        for each bin against the fraction of positive classes, on the
        y-axis.

        Extra keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.

        Read more about calibration in the :ref:`User Guide <calibration>` and
        more about the scikit-learn visualization API in :ref:`visualizations`.

        .. versionadded:: 1.0

        Parameters
        ----------
        estimator : estimator instance
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a classifier. The classifier must
            have a :term:`predict_proba` method.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.

        y : array-like of shape (n_samples,)
            Binary target values.

        n_bins : int, default=5
            Number of bins to discretize the [0, 1] interval into when
            calculating the calibration curve. A bigger number requires more
            data.

        strategy : {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.

            - `'uniform'`: The bins have identical widths.
            - `'quantile'`: The bins have the same number of samples and depend
              on predicted probabilities.

        pos_label : int, float, bool or str, default=None
            The positive class when computing the calibration curve.
            By default, `estimators.classes_[1]` is considered as the
            positive class.

            .. versionadded:: 1.1

        name : str, default=None
            Name for labeling curve. If `None`, the name of the estimator is
            used.

        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`.
            Object that stores computed values.

        See Also
        --------
        CalibrationDisplay.from_predictions : Plot calibration curve using true
            and predicted labels.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.calibration import CalibrationDisplay
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, random_state=0)
        >>> clf = LogisticRegression(random_state=0)
        >>> clf.fit(X_train, y_train)
        LogisticRegression(random_state=0)
        >>> disp = CalibrationDisplay.from_estimator(clf, X_test, y_test)
        >>> plt.show()
        """
        y_prob, pos_label, name = cls._validate_and_get_response_values(
            estimator,
            X,
            y,
            response_method="predict_proba",
            pos_label=pos_label,
            name=name,
        )

        return cls.from_predictions(
            y,
            y_prob,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=pos_label,
            name=name,
            ref_line=ref_line,
            ax=ax,
            **kwargs,
        )

def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        n_bins=5,
        strategy="uniform",
        pos_label=None,
        name=None,
        ref_line=True,
        ax=None,
        **kwargs,
    ):
        """Plot calibration curve using a binary classifier and data.

        A calibration curve, also known as a reliability diagram, uses inputs
        from a binary classifier and plots the average predicted probability
        for each bin against the fraction of positive classes, on the
        y-axis.

        Extra keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.

        Read more about calibration in the :ref:`User Guide <calibration>` and
        more about the scikit-learn visualization API in :ref:`visualizations`.

        .. versionadded:: 1.0

        Parameters
        ----------
        estimator : estimator instance
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a classifier. The classifier must
            have a :term:`predict_proba` method.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.

        y : array-like of shape (n_samples,)
            Binary target values.

        n_bins : int, default=5
            Number of bins to discretize the [0, 1] interval into when
            calculating the calibration curve. A bigger number requires more
            data.

        strategy : {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.

            - `'uniform'`: The bins have identical widths.
            - `'quantile'`: The bins have the same number of samples and depend
              on predicted probabilities.

        pos_label : int, float, bool or str, default=None
            The positive class when computing the calibration curve.
            By default, `estimators.classes_[1]` is considered as the
            positive class.

            .. versionadded:: 1.1

        name : str, default=None
            Name for labeling curve. If `None`, the name of the estimator is
            used.

        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`.
            Object that stores computed values.

        See Also
        --------
        CalibrationDisplay.from_predictions : Plot calibration curve using true
            and predicted labels.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.calibration import CalibrationDisplay
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, random_state=0)
        >>> clf = LogisticRegression(random_state=0)
        >>> clf.fit(X_train, y_train)
        LogisticRegression(random_state=0)
        >>> disp = CalibrationDisplay.from_estimator(clf, X_test, y_test)
        >>> plt.show()
        """
        y_prob, pos_label, name = cls._validate_and_get_response_values(
            estimator,
            X,
            y,
            response_method="predict_proba",
            pos_label=pos_label,
            name=name,
        )

        return cls.from_predictions(
            y,
            y_prob,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=pos_label,
            name=name,
            ref_line=ref_line,
            ax=ax,
            **kwargs,
        )

def calibration_curve(
    y_true,
    y_prob,
    *,
    pos_label=None,
    n_bins=5,
    strategy="uniform",
):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    pos_label : int, float, bool or str, default=None
        The label of the positive class.

        .. versionadded:: 1.1

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred

def _sigmoid_calibration(
    predictions, y, sample_weight=None, max_abs_prediction_threshold=30
):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    predictions : ndarray of shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray of shape (n_samples,)
        The targets.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    a : float
        The slope.

    b : float
        The intercept.

    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    predictions = column_or_1d(predictions)
    y = column_or_1d(y)

    F = predictions  # F follows Platt's notations

    scale_constant = 1.0
    max_prediction = np.max(np.abs(F))

    # If the predictions have large values we scale them in order to bring
    # them within a suitable range. This has no effect on the final
    # (prediction) result because linear models like Logisitic Regression
    # without a penalty are invariant to multiplying the features by a
    # constant.
    if max_prediction >= max_abs_prediction_threshold:
        scale_constant = max_prediction
        # We rescale the features in a copy: inplace rescaling could confuse
        # the caller and make the code harder to reason about.
        F = F / scale_constant

    # Bayesian priors (see Platt end of section 2.2):
    # It corresponds to the number of samples, taking into account the
    # `sample_weight`.
    mask_negative_samples = y <= 0
    if sample_weight is not None:
        prior0 = (sample_weight[mask_negative_samples]).sum()
        prior1 = (sample_weight[~mask_negative_samples]).sum()
    else:
        prior0 = float(np.sum(mask_negative_samples))
        prior1 = y.shape[0] - prior0
    T = np.zeros_like(y, dtype=predictions.dtype)
    T[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
    T[y <= 0] = 1.0 / (prior0 + 2.0)

    bin_loss = HalfBinomialLoss()

    def loss_grad(AB):
        # .astype below is needed to ensure y_true and raw_prediction have the
        # same dtype. With result = np.float64(0) * np.array([1, 2], dtype=np.float32)
        # - in Numpy 2, result.dtype is float64
        # - in Numpy<2, result.dtype is float32
        raw_prediction = -(AB[0] * F + AB[1]).astype(dtype=predictions.dtype)
        l, g = bin_loss.loss_gradient(
            y_true=T,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )
        loss = l.sum()
        # TODO: Remove casting to np.float64 when minimum supported SciPy is 1.11.2
        # With SciPy >= 1.11.2, the LBFGS implementation will cast to float64
        # https://github.com/scipy/scipy/pull/18825.
        # Here we cast to float64 to support SciPy < 1.11.2
        grad = np.asarray([-g @ F, -g.sum()], dtype=np.float64)
        return loss, grad

    AB0 = np.array([0.0, log((prior0 + 1.0) / (prior1 + 1.0))])

    opt_result = minimize(
        loss_grad,
        AB0,
        method="L-BFGS-B",
        jac=True,
        options={
            "gtol": 1e-6,
            "ftol": 64 * np.finfo(float).eps,
        },
    )
    AB_ = opt_result.x

    # The tuned multiplicative parameter is converted back to the original
    # input feature scale. The offset parameter does not need rescaling since
    # we did not rescale the outcome variable.
    return AB_[0] / scale_constant, AB_[1]

class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Probability calibration with isotonic regression or logistic regression.

    This class uses cross-validation to both estimate the parameters of a
    classifier and subsequently calibrate a classifier. With default
    `ensemble=True`, for each cv split it
    fits a copy of the base estimator to the training subset, and calibrates it
    using the testing subset. For prediction, predicted probabilities are
    averaged across these individual calibrated classifiers. When
    `ensemble=False`, cross-validation is used to obtain unbiased predictions,
    via :func:`~sklearn.model_selection.cross_val_predict`, which are then
    used for calibration. For prediction, the base estimator, trained using all
    the data, is used. This is the prediction method implemented when
    `probabilities=True` for :class:`~sklearn.svm.SVC` and :class:`~sklearn.svm.NuSVC`
    estimators (see :ref:`User Guide <scores_probabilities>` for details).

    Already fitted classifiers can be calibrated via the parameter
    `cv="prefit"`. In this case, no cross-validation is used and all provided
    data is used for calibration. The user has to take care manually that data
    for model fitting and calibration are disjoint.

    The calibration is based on the :term:`decision_function` method of the
    `estimator` if it exists, else on :term:`predict_proba`.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    estimator : estimator instance, default=None
        The classifier whose output need to be calibrated to provide more
        accurate `predict_proba` outputs. The default classifier is
        a :class:`~sklearn.svm.LinearSVC`.

        .. versionadded:: 1.2

    method : {'sigmoid', 'isotonic'}, default='sigmoid'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method (i.e. a logistic regression model) or
        'isotonic' which is a non-parametric approach. It is not advised to
        use isotonic calibration with too few calibration samples
        ``(<<1000)`` since it tends to overfit.

    cv : int, cross-validation generator, iterable or "prefit", \
            default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`~sklearn.model_selection.KFold`
        is used.

        Refer to the :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that `estimator` has been
        fitted already and all data is used for calibration.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

        Base estimator clones are fitted in parallel across cross-validation
        iterations. Therefore parallelism happens only when `cv != "prefit"`.

        See :term:`Glossary <n_jobs>` for more details.

        .. versionadded:: 0.24

    ensemble : bool, default=True
        Determines how the calibrator is fitted when `cv` is not `'prefit'`.
        Ignored if `cv='prefit'`.

        If `True`, the `estimator` is fitted using training data, and
        calibrated using testing data, for each `cv` fold. The final estimator
        is an ensemble of `n_cv` fitted classifier and calibrator pairs, where
        `n_cv` is the number of cross-validation folds. The output is the
        average predicted probabilities of all pairs.

        If `False`, `cv` is used to compute unbiased predictions, via
        :func:`~sklearn.model_selection.cross_val_predict`, which are then
        used for calibration. At prediction time, the classifier used is the
        `estimator` trained on all the data.
        Note that this method is also internally implemented  in
        :mod:`sklearn.svm` estimators with the `probabilities=True` parameter.

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    calibrated_classifiers_ : list (len() equal to cv or 1 if `cv="prefit"` \
            or `ensemble=False`)
        The list of classifier and calibrator pairs.

        - When `cv="prefit"`, the fitted `estimator` and fitted
          calibrator.
        - When `cv` is not "prefit" and `ensemble=True`, `n_cv` fitted
          `estimator` and calibrator pairs. `n_cv` is the number of
          cross-validation folds.
        - When `cv` is not "prefit" and `ensemble=False`, the `estimator`,
          fitted on all the data, and fitted calibrator.

        .. versionchanged:: 0.24
            Single calibrated classifier case when `ensemble=False`.

    See Also
    --------
    calibration_curve : Compute true and predicted probabilities
        for a calibration curve.

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.calibration import CalibratedClassifierCV
    >>> X, y = make_classification(n_samples=100, n_features=2,
    ...                            n_redundant=0, random_state=42)
    >>> base_clf = GaussianNB()
    >>> calibrated_clf = CalibratedClassifierCV(base_clf, cv=3)
    >>> calibrated_clf.fit(X, y)
    CalibratedClassifierCV(...)
    >>> len(calibrated_clf.calibrated_classifiers_)
    3
    >>> calibrated_clf.predict_proba(X)[:5, :]
    array([[0.110..., 0.889...],
           [0.072..., 0.927...],
           [0.928..., 0.071...],
           [0.928..., 0.071...],
           [0.071..., 0.928...]])
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, n_features=2,
    ...                            n_redundant=0, random_state=42)
    >>> X_train, X_calib, y_train, y_calib = train_test_split(
    ...        X, y, random_state=42
    ... )
    >>> base_clf = GaussianNB()
    >>> base_clf.fit(X_train, y_train)
    GaussianNB()
    >>> calibrated_clf = CalibratedClassifierCV(base_clf, cv="prefit")
    >>> calibrated_clf.fit(X_calib, y_calib)
    CalibratedClassifierCV(...)
    >>> len(calibrated_clf.calibrated_classifiers_)
    1
    >>> calibrated_clf.predict_proba([[-0.5, 0.5]])
    array([[0.936..., 0.063...]])
    """

    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
            None,
        ],
        "method": [StrOptions({"isotonic", "sigmoid"})],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "n_jobs": [Integral, None],
        "ensemble": ["boolean"],
    }

    def __init__(
        self,
        estimator=None,
        *,
        method="sigmoid",
        cv=None,
        n_jobs=None,
        ensemble=True,
    ):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.ensemble = ensemble

    def _get_estimator(self):
        """Resolve which estimator to return (default is LinearSVC)"""
        if self.estimator is None:
            # we want all classifiers that don't expose a random_state
            # to be deterministic (and we don't want to expose this one).
            estimator = LinearSVC(random_state=0)
            if _routing_enabled():
                estimator.set_fit_request(sample_weight=True)
        else:
            estimator = self.estimator

        return estimator

    @_fit_context(
        # CalibratedClassifierCV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the calibrated model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        **fit_params : dict
            Parameters to pass to the `fit` method of the underlying
            classifier.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        check_classification_targets(y)
        X, y = indexable(X, y)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        estimator = self._get_estimator()

        self.calibrated_classifiers_ = []
        if self.cv == "prefit":
            # `classes_` should be consistent with that of estimator
            check_is_fitted(self.estimator, attributes=["classes_"])
            self.classes_ = self.estimator.classes_

            predictions, _ = _get_response_values(
                estimator,
                X,
                response_method=["decision_function", "predict_proba"],
            )
            if predictions.ndim == 1:
                # Reshape binary output from `(n_samples,)` to `(n_samples, 1)`
                predictions = predictions.reshape(-1, 1)

            calibrated_classifier = _fit_calibrator(
                estimator,
                predictions,
                y,
                self.classes_,
                self.method,
                sample_weight,
            )
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            # Set `classes_` using all `y`
            label_encoder_ = LabelEncoder().fit(y)
            self.classes_ = label_encoder_.classes_

            if _routing_enabled():
                routed_params = process_routing(
                    self,
                    "fit",
                    sample_weight=sample_weight,
                    **fit_params,
                )
            else:
                # sample_weight checks
                fit_parameters = signature(estimator.fit).parameters
                supports_sw = "sample_weight" in fit_parameters
                if sample_weight is not None and not supports_sw:
                    estimator_name = type(estimator).__name__
                    warnings.warn(
                        f"Since {estimator_name} does not appear to accept"
                        " sample_weight, sample weights will only be used for the"
                        " calibration itself. This can be caused by a limitation of"
                        " the current scikit-learn API. See the following issue for"
                        " more details:"
                        " https://github.com/scikit-learn/scikit-learn/issues/21134."
                        " Be warned that the result of the calibration is likely to be"
                        " incorrect."
                    )
                routed_params = Bunch()
                routed_params.splitter = Bunch(split={})  # no routing for splitter
                routed_params.estimator = Bunch(fit=fit_params)
                if sample_weight is not None and supports_sw:
                    routed_params.estimator.fit["sample_weight"] = sample_weight

            # Check that each cross-validation fold can have at least one
            # example per class
            if isinstance(self.cv, int):
                n_folds = self.cv
            elif hasattr(self.cv, "n_splits"):
                n_folds = self.cv.n_splits
            else:
                n_folds = None
            if n_folds and np.any(np.unique(y, return_counts=True)[1] < n_folds):
                raise ValueError(
                    f"Requesting {n_folds}-fold "
                    "cross-validation but provided less than "
                    f"{n_folds} examples for at least one class."
                )
            cv = check_cv(self.cv, y, classifier=True)

            if self.ensemble:
                parallel = Parallel(n_jobs=self.n_jobs)
                self.calibrated_classifiers_ = parallel(
                    delayed(_fit_classifier_calibrator_pair)(
                        clone(estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        method=self.method,
                        classes=self.classes_,
                        sample_weight=sample_weight,
                        fit_params=routed_params.estimator.fit,
                    )
                    for train, test in cv.split(X, y, **routed_params.splitter.split)
                )
            else:
                this_estimator = clone(estimator)
                method_name = _check_response_method(
                    this_estimator,
                    ["decision_function", "predict_proba"],
                ).__name__
                predictions = cross_val_predict(
                    estimator=this_estimator,
                    X=X,
                    y=y,
                    cv=cv,
                    method=method_name,
                    n_jobs=self.n_jobs,
                    params=routed_params.estimator.fit,
                )
                if len(self.classes_) == 2:
                    # Ensure shape (n_samples, 1) in the binary case
                    if method_name == "predict_proba":
                        # Select the probability column of the postive class
                        predictions = _process_predict_proba(
                            y_pred=predictions,
                            target_type="binary",
                            classes=self.classes_,
                            pos_label=self.classes_[1],
                        )
                    predictions = predictions.reshape(-1, 1)

                this_estimator.fit(X, y, **routed_params.estimator.fit)
                # Note: Here we don't pass on fit_params because the supported
                # calibrators don't support fit_params anyway
                calibrated_classifier = _fit_calibrator(
                    this_estimator,
                    predictions,
                    y,
                    self.classes_,
                    self.method,
                    sample_weight,
                )
                self.calibrated_classifiers_.append(calibrated_classifier)

        first_clf = self.calibrated_classifiers_[0].estimator
        if hasattr(first_clf, "n_features_in_"):
            self.n_features_in_ = first_clf.n_features_in_
        if hasattr(first_clf, "feature_names_in_"):
            self.feature_names_in_ = first_clf.feature_names_in_
        return self

    def predict_proba(self, X):
        """Calibrated probabilities of classification.

        This function returns calibrated probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict_proba`.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            The predicted probas.
        """
        check_is_fitted(self)
        # Compute the arithmetic mean of the predictions of the calibrated
        # classifiers
        mean_proba = np.zeros((_num_samples(X), len(self.classes_)))
        for calibrated_classifier in self.calibrated_classifiers_:
            proba = calibrated_classifier.predict_proba(X)
            mean_proba += proba

        mean_proba /= len(self.calibrated_classifiers_)

        return mean_proba

    def predict(self, X):
        """Predict the target of new samples.

        The predicted class is the class that has the highest probability,
        and can thus be different from the prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict`.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                estimator=self._get_estimator(),
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                splitter=self.cv,
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
        )
        return router

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "Due to the cross-validation and sample ordering, removing a sample"
                    " is not strictly equal to putting is weight to zero. Specific unit"
                    " tests are added for CalibratedClassifierCV specifically."
                ),
            }
        }

