class DeadZoneRegressor(BaseEstimator, RegressorMixin):
    r"""The `DeadZoneRegressor` estimator implements a regression model that incorporates a _dead zone effect_ for
    improving the robustness of regression predictions.

    The dead zone effect allows the model to reduce the impact of small errors in the training data on the regression
    results, which can be particularly useful when dealing with noisy or unreliable data.

    The estimator minimizes the following loss function using gradient descent:

    $$\frac{1}{n} \sum_{i=1}^{n} \text{deadzone}\left(\left|X_i \cdot w - y_i\right|\right)$$

    where:

    $$\text{deadzone}(e) =
    \begin{cases}
    1 & \text{if } e > \text{threshold} \text{ & effect="constant"} \\
    e & \text{if } e > \text{threshold} \text{ & effect="linear"} \\
    e^2 & \text{if } e > \text{threshold} \text{ & effect="quadratic"} \\
    0 & \text{otherwise}
    \end{cases}
    $$

    Parameters
    ----------
    threshold : float, default=0.3
        The threshold value for the dead zone effect.
    relative : bool, default=False
        If True, the threshold is relative to the target value. Namely the _dead zone effect_ is applied to the
        relative error between the predicted and target values.
    effect : Literal["linear", "quadratic", "constant"], default="linear"
        The type of dead zone effect to apply. It can be one of the following:

        - "linear": the errors within the threshold have no impact (their contribution is effectively zero), and errors
            outside the threshold are penalized linearly.
        - "quadratic": the errors within the threshold have no impact (their contribution is effectively zero), and
            errors outside the threshold are penalized quadratically (squared).
        - "constant": the errors within the threshold have no impact, and errors outside the threshold are penalized
            with a constant value.
    n_iter : int, default=2000
        The number of iterations to run the gradient descent algorithm.
    stepsize : float, default=0.01
        The step size for the gradient descent algorithm.
    check_grad : bool, default=False
        If True, check the gradients numerically, _just to be safe_.

    Attributes
    ----------
    coef_ : np.ndarray, shape (n_columns,)
        The learned coefficients after fitting the model.
    coefs_ : np.ndarray, shape (n_columns,)
        Deprecated, please use `coef_` instead.
    """

    _ALLOWED_EFFECTS = ("linear", "quadratic", "constant")

    def __init__(
        self,
        threshold=0.3,
        relative=False,
        effect="linear",
    ):
        self.threshold = threshold
        self.relative = relative
        self.effect = effect

    def fit(self, X, y):
        """Fit the estimator on training data `X` and `y` by optimizing the loss function using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : DeadZoneRegressor
            The fitted estimator.

        Raises
        ------
        ValueError
            If `effect` is not one of "linear", "quadratic" or "constant".
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        if self.effect not in self._ALLOWED_EFFECTS:
            raise ValueError(f"effect {self.effect} must be in {self._ALLOWED_EFFECTS}")

        def deadzone(errors):
            if self.effect == "constant":
                error_weight = errors.shape[0]
            elif self.effect == "linear":
                error_weight = errors
            elif self.effect == "quadratic":
                error_weight = errors**2

            return np.where(errors > self.threshold, error_weight, 0.0)

        def training_loss(weights):
            prediction = np.dot(X, weights)
            errors = np.abs(prediction - y)

            if self.relative:
                errors /= np.abs(y)

            loss = np.mean(deadzone(errors))
            return loss

        def deadzone_derivative(errors):
            if self.effect == "constant":
                error_weight = 0.0
            elif self.effect == "linear":
                error_weight = 1.0
            elif self.effect == "quadratic":
                error_weight = 2 * errors

            return np.where(errors > self.threshold, error_weight, 0.0)

        def training_loss_derivative(weights):
            prediction = np.dot(X, weights)
            errors = np.abs(prediction - y)

            if self.relative:
                errors /= np.abs(y)

            loss_derivative = deadzone_derivative(errors)
            errors_derivative = np.sign(prediction - y)

            if self.relative:
                errors_derivative /= np.abs(y)

            derivative = np.dot(errors_derivative * loss_derivative, X) / X.shape[0]

            return derivative

        self.n_features_in_ = X.shape[1]

        minimize_result = minimize(
            training_loss,
            x0=np.zeros(self.n_features_in_),  # np.random.normal(0, 1, n_features_)
            tol=1e-20,
            jac=training_loss_derivative,
        )

        self.convergence_status_ = minimize_result.message
        self.coef_ = minimize_result.x
        return self

    def predict(self, X):
        """Predict target values for `X` using fitted estimator by multiplying `X` with the learned coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["coef_"])
        return np.dot(X, self.coef_)

    @property
    def coefs_(self):
        warn(
            "Please use `coef_` instead of `coefs_`, `coefs_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.coef_

    @property
    def allowed_effects(self):
        warn(
            "Please use `_ALLOWED_EFFECTS` instead of `allowed_effects`,"
            "`allowed_effects` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self._ALLOWED_EFFECTS

class ImbalancedLinearRegression(BaseScipyMinimizeRegressor):
    r"""Linear regression where overestimating is `overestimation_punishment_factor` times worse than underestimating.

    A value of `overestimation_punishment_factor=5` implies that overestimations by the model are penalized with a
    factor of 5 while underestimations have a default factor of 1. The formula optimized for is

    $$\frac{1}{2 N} \|s \circ (y - Xw) \|_2^2 + \alpha \cdot l_1 \cdot\|w\|_1 + \frac{\alpha}{2} \cdot (1-l_1)\cdot
    \|w\|_2^2$$

    where $\circ$ is component-wise multiplication and

    $$ s = \begin{cases}
    \text{overestimation_punishment_factor} & \text{if } y - Xw < 0 \\
    1 & \text{otherwise}
    \end{cases}
    $$

    `ImbalancedLinearRegression` fits a linear model to minimize the residual sum of squares between the observed
    targets in the dataset, and the targets predicted by the linear approximation.
    Compared to normal linear regression, this approach allows for a different treatment of over or under estimations.

    !!! info
        This implementation uses
        [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms.
    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`:

        - `l1_ratio = 0` is equivalent to an L2 penalty.
        - `l1_ratio = 1` is equivalent to an L1 penalty.
        - `0 < l1_ratio < 1` is the combination of L1 and L2.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    copy_X : bool, default=True
        If True, `X` will be copied; else, it may be overwritten.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    method : Literal["SLSQP", "TNC", "L-BFGS-B"], default="SLSQP"
        Type of solver to use for optimization.
    overestimation_punishment_factor : float, default=1.0
        Factor to punish overestimations more (if the value is larger than 1) or less (if the value is between 0 and 1).

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients of the model.
    intercept_ : float
        Independent term in the linear model. Set to 0.0 if `fit_intercept = False`.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    ```py
    import numpy as np
    from sklego.linear_model import ImbalancedLinearRegression

    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = X @ np.array([1, 2, 3, 4]) + 2*np.random.randn(100)

    over_bad = ImbalancedLinearRegression(overestimation_punishment_factor=50).fit(X, y)
    over_bad.coef_
    # array([0.36267036, 1.39526844, 3.4247146 , 3.93679175])

    under_bad = ImbalancedLinearRegression(overestimation_punishment_factor=0.01).fit(X, y)
    under_bad.coef_
    # array([0.73519586, 1.28698197, 2.61362614, 4.35989806])
    ```
    """

    def __init__(
        self,
        alpha=0.0,
        l1_ratio=0.0,
        fit_intercept=True,
        copy_X=True,
        positive=False,
        method="SLSQP",
        overestimation_punishment_factor=1.0,
    ):
        super().__init__(alpha, l1_ratio, fit_intercept, copy_X, positive, method)
        self.overestimation_punishment_factor = overestimation_punishment_factor

    def _get_objective(self, X, y, sample_weight):
        def imbalanced_loss(params):
            return 0.5 * np.average(
                np.where(X @ params > y, self.overestimation_punishment_factor, 1) * np.square(y - X @ params),
                weights=sample_weight,
            ) + self._regularized_loss(params)

        def grad_imbalanced_loss(params):
            return (
                -(sample_weight * np.where(X @ params > y, self.overestimation_punishment_factor, 1) * (y - X @ params))
                @ X
                / sample_weight.sum()
            ) + self._regularized_grad_loss(params)

        return imbalanced_loss, grad_imbalanced_loss

class QuantileRegression(BaseScipyMinimizeRegressor):
    r"""Compute quantile regression. This can be used for computing confidence intervals of linear regressions.

    `QuantileRegression` fits a linear model to minimize a weighted residual sum of absolute deviations between
    the observed targets in the dataset and the targets predicted by the linear approximation, i.e.

    $$\frac{\text{switch} \cdot ||y - Xw||_1}{2 N} + \alpha \cdot l_1 \cdot ||w||_1
        + \frac{\alpha}{2} \cdot (1 - l_1) \cdot ||w||^2_2$$

    where

    $$\text{switch} = \begin{cases}
    \text{quantile} & \text{if } y - Xw < 0 \\
    1-\text{quantile} & \text{otherwise}
    \end{cases}$$

    The regressor defaults to `LADRegression` for its default value of `quantile=0.5`.

    Compared to linear regression, this approach is robust to outliers.

    !!! info
        This implementation uses
        [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

    !!! warning
        If, while fitting the model, `sample_weight` contains any zero values, some solvers may not converge properly.
        We would expect that a sample weight of zero is equivalent to removing the sample, however unittests tell us
        that this is always the case only for `method='SLSQP'` (our default)

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms.
    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`:

        - `l1_ratio = 0` is equivalent to an L2 penalty.
        - `l1_ratio = 1` is equivalent to an L1 penalty.
        - `0 < l1_ratio < 1` is the combination of L1 and L2.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    copy_X : bool, default=True
        If True, `X` will be copied; else, it may be overwritten.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    method : Literal["SLSQP", "TNC", "L-BFGS-B"], default="SLSQP"
        Type of solver to use for optimization.
    quantile : float, default=0.5
        The line output by the model will have a share of approximately `quantile` data points under it. It  should
        be a value between 0 and 1.

        A value of `quantile=1` outputs a line that is above each data point, for example.
        `quantile=0.5` corresponds to LADRegression.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients of the model.
    intercept_ : float
        Independent term in the linear model. Set to 0.0 if `fit_intercept = False`.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    ```py
    import numpy as np
    from sklego.linear_model import QuantileRegression

    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = X @ np.array([1, 2, 3, 4])

    model = QuantileRegression().fit(X, y)
    model.coef_
    # array([1., 2., 3., 4.])

    y = X @ np.array([-1, 2, -3, 4])
    model = QuantileRegression(quantile=0.8).fit(X, y)
    model.coef_
    # array([-1.,  2., -3.,  4.])
    ```
    """

    def __init__(
        self,
        alpha=0.0,
        l1_ratio=0.0,
        fit_intercept=True,
        copy_X=True,
        positive=False,
        method="SLSQP",
        quantile=0.5,
    ):
        super().__init__(alpha, l1_ratio, fit_intercept, copy_X, positive, method)
        self.quantile = quantile

    def _get_objective(self, X, y, sample_weight):
        def quantile_loss(params):
            return np.average(
                np.where(X @ params < y, self.quantile, 1 - self.quantile) * np.abs(y - X @ params),
                weights=sample_weight,
            ) + self._regularized_loss(params)

        def grad_quantile_loss(params):
            return (
                -(sample_weight * np.where(X @ params < y, self.quantile, 1 - self.quantile) * np.sign(y - X @ params))
                @ X
                / sample_weight.sum()
            ) + self._regularized_grad_loss(params)

        return quantile_loss, grad_quantile_loss

    def fit(self, X, y, sample_weight=None):
        """Fit the estimator on training data `X` and `y` by minimizing the quantile loss function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,) | None, default=None
            Individual weights for each sample.

        Returns
        -------
        self : QuantileRegression
            The fitted estimator.

        Raises
        ------
        ValueError
            If `quantile` is not between 0 and 1.
        """
        if 0 <= self.quantile <= 1:
            super().fit(X, y, sample_weight)
        else:
            raise ValueError("Parameter `quantile` should be between zero and one.")

        return self

class EqualOpportunityClassifier(BaseEstimator, LinearClassifierMixin):
    r"""`EqualOpportunityClassifier` is a logistic regression classifier which can be constrained on equal opportunity
    score.

    It minimizes the log loss while constraining the correlation between the specified `sensitive_cols` and the
    distance to the decision boundary of the classifier for those examples that have a y_true of 1.

    !!! warning
        This classifier only works for binary classification problems.

    $$\begin{array}{cl}{\operatorname{minimize}} & -\sum_{i=1}^{N} \log p\left(y_{i} | \mathbf{x}_{i},
        \boldsymbol{\theta}\right) \\
        {\text { subject to }} & {\frac{1}{POS} \sum_{i=1}^{POS}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_\boldsymbol{\theta}\left(\mathbf{x}_{i}\right) \leq \mathbf{c}} \\
        {} & {\frac{1}{POS} \sum_{i=1}^{POS}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right) \geq-\mathbf{c}}\end{array}$$

    where POS is the subset of the population where $\text{y_true} = 1$

    Parameters
    ----------
    covariance_threshold : float | None
        The maximum allowed covariance between the sensitive attributes and the distance to the decision boundary.
        If set to None, no fairness constraint is enforced.
    positive_target : int
        The name of the class which is associated with a positive outcome
    sensitive_cols : List[str] | List[int] | None, default=None
        List of sensitive column names (if X is a dataframe) or a list of column indices (if X is a numpy array).
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values
        specify stronger regularization.
    penalty : Literal["l1", "l2", "none", None], default="l1"
        The type of penalty to apply to the model. "l1" applies L1 regularization, "l2" applies L2 regularization,
        while None (or "none") disables regularization.
    fit_intercept : bool, default=True
        Whether or not a constant term (a.k.a. bias or intercept) should be added to the decision function.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    train_sensitive_cols : bool, default=False
        Indicates whether the model should use the sensitive columns in the fit step.
    multi_class : Literal["ovr", "ovo"], default="ovr"
        The method to use for multiclass predictions.
    n_jobs : int | None, default=1
        The amount of parallel jobs that should be used to fit the model.
    """

    def __new__(cls, *args, multi_class="ovr", n_jobs=1, **kwargs):
        multiclass_meta = {"ovr": OneVsRestClassifier, "ovo": OneVsOneClassifier}[multi_class]
        return multiclass_meta(_EqualOpportunityClassifier(*args, **kwargs), n_jobs=n_jobs)

class DemographicParityClassifier(BaseEstimator, LinearClassifierMixin):
    r"""`DemographicParityClassifier` is a logistic regression classifier which can be constrained on demographic
    parity (p% score).

    It minimizes the log loss while constraining the correlation between the specified `sensitive_cols` and the
    distance to the decision boundary of the classifier.

    !!! warning
        This classifier only works for binary classification problems.

    $$\begin{array}{cl}{\operatorname{minimize}} & -\sum_{i=1}^{N} \log p\left(y_{i} | \mathbf{x}_{i},
        \boldsymbol{\theta}\right) \\
        {\text { subject to }} & {\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_\boldsymbol{\theta}\left(\mathbf{x}_{i}\right) \leq \mathbf{c}} \\
        {} & {\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right) \geq-\mathbf{c}}\end{array}$$

    Parameters
    ----------
    covariance_threshold : float | None
        The maximum allowed covariance between the sensitive attributes and the distance to the decision boundary.
        If set to None, no fairness constraint is enforced.
    sensitive_cols : List[str] | List[int] | None, default=None
        List of sensitive column names (if X is a dataframe) or a list of column indices (if X is a numpy array).
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values
        specify stronger regularization.
    penalty : Literal["l1", "l2", "none", None], default="l1"
        The type of penalty to apply to the model. "l1" applies L1 regularization, "l2" applies L2 regularization,
        while None (or "none") disables regularization.
    fit_intercept : bool, default=True
        Whether or not a constant term (a.k.a. bias or intercept) should be added to the decision function.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    train_sensitive_cols : bool, default=False
        Indicates whether the model should use the sensitive columns in the fit step.
    multi_class : Literal["ovr", "ovo"], default="ovr"
        The method to use for multiclass predictions.
    n_jobs : int | None, default=1
        The amount of parallel jobs that should be used to fit the model.

    Source
    ------
    M. Zafar et al. (2017), Fairness Constraints: Mechanisms for Fair Classification
    """

    def __new__(cls, *args, multi_class="ovr", n_jobs=1, **kwargs):
        multiclass_meta = {"ovr": OneVsRestClassifier, "ovo": OneVsOneClassifier}[multi_class]
        return multiclass_meta(_DemographicParityClassifier(*args, **kwargs), n_jobs=n_jobs)

class ProbWeightRegression(BaseEstimator, RegressorMixin):
    """`ProbWeightRegression` assumes that all input signals in `X` need to be reweighted with weights that sum up to
    one in order to predict `y`.

    This can be very useful in combination with `sklego.meta.EstimatorTransformer` because it allows to construct
    an ensemble.

    Parameters
    ----------
    non_negative : bool, default=True
        If True, forces all weights to be non-negative.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.
    coef_ : np.ndarray, shape (n_columns,)
        The learned coefficients after fitting the model.
    coefs_ : np.ndarray, shape (n_columns,)
        Deprecated, please use `coef_` instead.

    !!! info

        This model requires [`cvxpy`](https://www.cvxpy.org/) to be installed. If you don't have it installed, you can
        install it with:

        ```bash
        pip install cvxpy
        # or pip install scikit-lego"[cvxpy]"
        ```
    """

    def __init__(self, non_negative=True):
        self.non_negative = non_negative

    def fit(self, X, y):
        r"""Fit the estimator on training data `X` and `y` by solving the following convex optimization problem:

        $$\begin{array}{ll}{\operatorname{minimize}} & {\sum_{i=1}^{N}\left(\mathbf{x}_{i}
        \boldsymbol{\beta}-y_{i}\right)^{2}} \\
        {\text { subject to }} & {\sum_{j=1}^{p} \beta_{j}=1} \\
        {(\text{If non_negative=True})} & {\beta_{j} \geq 0, \quad j=1, \ldots, p} \end{array}$$

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : ProbWeightRegression
            The fitted estimator.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)

        # Construct the problem.
        betas = cp.Variable(X.shape[1])
        objective = cp.Minimize(cp.sum_squares(X @ betas - y))
        constraints = [sum(betas) == 1]
        if self.non_negative:
            constraints.append(0 <= betas)

        # Solve the problem.
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.coef_ = betas.value
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict target values for `X` using fitted estimator by multiplying `X` with the learned coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["coef_"])
        return np.dot(X, self.coef_)

    @property
    def coefs_(self):
        warn(
            "Please use `coef_` instead of `coefs_`, `coefs_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.coef_

