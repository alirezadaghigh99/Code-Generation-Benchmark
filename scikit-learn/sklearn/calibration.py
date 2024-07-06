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

