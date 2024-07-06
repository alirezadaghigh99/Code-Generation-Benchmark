def scale(X, *, axis=0, with_mean=True, with_std=True, copy=True):
    """Standardize a dataset along any axis.

    Center to the mean and component wise scale to unit variance.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to center and scale.

    axis : {0, 1}, default=0
        Axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    StandardScaler : Performs scaling to unit variance using the Transformer
        API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the CSC matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSC matrix.

    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.StandardScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(StandardScaler(), LogisticRegression())`.

    Examples
    --------
    >>> from sklearn.preprocessing import scale
    >>> X = [[-2, 1, 2], [-1, 0, 1]]
    >>> scale(X, axis=0)  # scaling each column independently
    array([[-1.,  1.,  1.],
           [ 1., -1., -1.]])
    >>> scale(X, axis=1)  # scaling each row independently
    array([[-1.37...,  0.39...,  0.98...],
           [-1.22...,  0.     ,  1.22...]])
    """
    X = check_array(
        X,
        accept_sparse="csc",
        copy=copy,
        ensure_2d=False,
        estimator="the scale function",
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
    )
    if sparse.issparse(X):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead"
                " See docstring for motivation and alternatives."
            )
        if axis != 0:
            raise ValueError(
                "Can only scale sparse matrix on axis=0,  got axis=%d" % axis
            )
        if with_std:
            _, var = mean_variance_axis(X, axis=0)
            var = _handle_zeros_in_scale(var, copy=False)
            inplace_column_scale(X, 1 / np.sqrt(var))
    else:
        X = np.asarray(X)
        if with_mean:
            mean_ = np.nanmean(X, axis)
        if with_std:
            scale_ = np.nanstd(X, axis)
        # Xr is a view on the original array that enables easy use of
        # broadcasting on the axis in which we are interested in
        Xr = np.rollaxis(X, axis)
        if with_mean:
            Xr -= mean_
            mean_1 = np.nanmean(Xr, axis=0)
            # Verify that mean_1 is 'close to zero'. If X contains very
            # large values, mean_1 can also be very large, due to a lack of
            # precision of mean_. In this case, a pre-scaling of the
            # concerned feature is efficient, for instance by its mean or
            # maximum.
            if not np.allclose(mean_1, 0):
                warnings.warn(
                    "Numerical issues were encountered "
                    "when centering the data "
                    "and might not be solved. Dataset may "
                    "contain too large values. You may need "
                    "to prescale your features."
                )
                Xr -= mean_1
        if with_std:
            scale_ = _handle_zeros_in_scale(scale_, copy=False)
            Xr /= scale_
            if with_mean:
                mean_2 = np.nanmean(Xr, axis=0)
                # If mean_2 is not 'close to zero', it comes from the fact that
                # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                # if mean_1 was close to zero. The problem is thus essentially
                # due to the lack of precision of mean_. A solution is then to
                # subtract the mean again:
                if not np.allclose(mean_2, 0):
                    warnings.warn(
                        "Numerical issues were encountered "
                        "when scaling the data "
                        "and might not be solved. The standard "
                        "deviation of the data is probably "
                        "very close to 0. "
                    )
                    Xr -= mean_2
    return X

