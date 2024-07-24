def init_gradient_and_hessian(self, n_samples, dtype=np.float64, order="F"):
        """Initialize arrays for gradients and hessians.

        Unless hessians are constant, arrays are initialized with undefined values.

        Parameters
        ----------
        n_samples : int
            The number of samples, usually passed to `fit()`.
        dtype : {np.float64, np.float32}, default=np.float64
            The dtype of the arrays gradient and hessian.
        order : {'C', 'F'}, default='F'
            Order of the arrays gradient and hessian. The default 'F' makes the arrays
            contiguous along samples.

        Returns
        -------
        gradient : C-contiguous array of shape (n_samples,) or array of shape \
            (n_samples, n_classes)
            Empty array (allocated but not initialized) to be used as argument
            gradient_out.
        hessian : C-contiguous array of shape (n_samples,), array of shape
            (n_samples, n_classes) or shape (1,)
            Empty (allocated but not initialized) array to be used as argument
            hessian_out.
            If constant_hessian is True (e.g. `HalfSquaredError`), the array is
            initialized to ``1``.
        """
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                "Valid options for 'dtype' are np.float32 and np.float64. "
                f"Got dtype={dtype} instead."
            )

        if self.is_multiclass:
            shape = (n_samples, self.n_classes)
        else:
            shape = (n_samples,)
        gradient = np.empty(shape=shape, dtype=dtype, order=order)

        if self.constant_hessian:
            # If the hessians are constant, we consider them equal to 1.
            # - This is correct for HalfSquaredError
            # - For AbsoluteError, hessians are actually 0, but they are
            #   always ignored anyway.
            hessian = np.ones(shape=(1,), dtype=dtype)
        else:
            hessian = np.empty(shape=shape, dtype=dtype, order=order)

        return gradient, hessian

class HalfBinomialLoss(BaseLoss):
    """Half Binomial deviance loss with logit link, for binary classification.

    This is also know as binary cross entropy, log-loss and logistic loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(raw_prediction)

    For a given sample x_i, half Binomial deviance is defined as the negative
    log-likelihood of the Binomial/Bernoulli distribution and can be expressed
    as::

        loss(x_i) = log(1 + exp(raw_pred_i)) - y_true_i * raw_pred_i

    See The Elements of Statistical Learning, by Hastie, Tibshirani, Friedman,
    section 4.4.1 (about logistic regression).

    Note that the formulation works for classification, y = {0, 1}, as well as
    logistic regression, y = [0, 1].
    If you add `constant_to_optimal_zero` to the loss, you get half the
    Bernoulli/binomial deviance.

    More details: Inserting the predicted probability y_pred = expit(raw_prediction)
    in the loss gives the well known::

        loss(x_i) = - y_true_i * log(y_pred_i) - (1 - y_true_i) * log(1 - y_pred_i)
    """

    def __init__(self, sample_weight=None):
        super().__init__(
            closs=CyHalfBinomialLoss(),
            link=LogitLink(),
            n_classes=2,
        )
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # This is non-zero only if y_true is neither 0 nor 1.
        term = xlogy(y_true, y_true) + xlogy(1 - y_true, 1 - y_true)
        if sample_weight is not None:
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        proba[:, 1] = self.link.inverse(raw_prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba

class HalfSquaredError(BaseLoss):
    """Half squared error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, half squared error is defined as::

        loss(x_i) = 0.5 * (y_true_i - raw_prediction_i)**2

    The factor of 0.5 simplifies the computation of gradients and results in a
    unit hessian (and is consistent with what is done in LightGBM). It is also
    half the Normal distribution deviance.
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyHalfSquaredError(), link=IdentityLink())
        self.constant_hessian = sample_weight is None

class PinballLoss(BaseLoss):
    """Quantile loss aka pinball loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the pinball loss is defined as::

        loss(x_i) = rho_{quantile}(y_true_i - raw_prediction_i)

        rho_{quantile}(u) = u * (quantile - 1_{u<0})
                          = -u *(1 - quantile)  if u < 0
                             u * quantile       if u >= 0

    Note: 2 * PinballLoss(quantile=0.5) equals AbsoluteError().

    Note that the exact hessian = 0 almost everywhere (except at one point, therefore
    differentiable = False). Optimization routines like in HGBT, however, need a
    hessian > 0. Therefore, we assign 1.

    Additional Attributes
    ---------------------
    quantile : float
        The quantile level of the quantile to be estimated. Must be in range (0, 1).
    """

    differentiable = False
    need_update_leaves_values = True

    def __init__(self, sample_weight=None, quantile=0.5):
        check_scalar(
            quantile,
            "quantile",
            target_type=numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )
        super().__init__(
            closs=CyPinballLoss(quantile=float(quantile)),
            link=IdentityLink(),
        )
        self.approx_hessian = True
        self.constant_hessian = sample_weight is None

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            return np.percentile(y_true, 100 * self.closs.quantile, axis=0)
        else:
            return _weighted_percentile(
                y_true, sample_weight, 100 * self.closs.quantile
            )

