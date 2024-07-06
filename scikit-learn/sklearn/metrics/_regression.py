def mean_squared_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    squared="deprecated",
):
    """Mean squared error regression loss.

    Read more in the :ref:`User Guide <mean_squared_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    squared : bool, default=True
        If True returns MSE value, if False returns RMSE value.

        .. deprecated:: 1.4
           `squared` is deprecated in 1.4 and will be removed in 1.6.
           Use :func:`~sklearn.metrics.root_mean_squared_error`
           instead to calculate the root mean squared error.

    Returns
    -------
    loss : float or array of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)
    0.708...
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.41666667, 1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.825...
    """
    # TODO(1.6): remove
    if squared != "deprecated":
        warnings.warn(
            (
                "'squared' is deprecated in version 1.4 and "
                "will be removed in 1.6. To calculate the "
                "root mean squared error, use the function"
                "'root_mean_squared_error'."
            ),
            FutureWarning,
        )
        if not squared:
            return root_mean_squared_error(
                y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
            )

    xp, _ = get_namespace(y_true, y_pred, sample_weight, multioutput)
    dtype = _find_matching_floating_dtype(y_true, y_pred, xp=xp)

    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput, dtype=dtype, xp=xp
    )
    check_consistent_length(y_true, y_pred, sample_weight)
    output_errors = _average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    # See comment in mean_absolute_error
    mean_squared_error = _average(output_errors, weights=multioutput)
    assert mean_squared_error.shape == ()
    return float(mean_squared_error)

def mean_pinball_loss(
    y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average"
):
    """Pinball loss for quantile regression.

    Read more in the :ref:`User Guide <pinball_loss>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    alpha : float, slope of the pinball loss, default=0.5,
        This loss is equivalent to :ref:`mean_absolute_error` when `alpha=0.5`,
        `alpha=0.95` is minimized by estimators of the 95th percentile.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.

        The pinball loss output is a non-negative floating point. The best
        value is 0.0.

    Examples
    --------
    >>> from sklearn.metrics import mean_pinball_loss
    >>> y_true = [1, 2, 3]
    >>> mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
    0.03...
    >>> mean_pinball_loss(y_true, [1, 2, 4], alpha=0.1)
    0.3...
    >>> mean_pinball_loss(y_true, [0, 2, 3], alpha=0.9)
    0.3...
    >>> mean_pinball_loss(y_true, [1, 2, 4], alpha=0.9)
    0.03...
    >>> mean_pinball_loss(y_true, y_true, alpha=0.1)
    0.0
    >>> mean_pinball_loss(y_true, y_true, alpha=0.9)
    0.0
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    check_consistent_length(y_true, y_pred, sample_weight)
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    output_errors = np.average(loss, weights=sample_weight, axis=0)

    if isinstance(multioutput, str) and multioutput == "raw_values":
        return output_errors

    if isinstance(multioutput, str) and multioutput == "uniform_average":
        # pass None as weights to np.average: uniform mean
        multioutput = None

    return np.average(output_errors, weights=multioutput)

