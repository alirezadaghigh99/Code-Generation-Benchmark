class ExpectedImprovement(Acquisition):
    def __init__(self, model: Union[IModel, IDifferentiable], jitter: float = 0.0) -> None:
        """
        For a given input, this acquisition computes the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        mean, variance = self._get_model_predictions(x)
        standard_deviation = np.sqrt(variance)
        mean += self.jitter

        y_minimum = self._get_y_minimum()
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)
        improvement = standard_deviation * (u * cdf + pdf)

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self._get_model_predictions(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = self._get_y_minimum()

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)

        improvement = standard_deviation * (u * cdf + pdf)
        dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx

        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)

    def _get_model_predictions(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions for the function values at given input locations."""
        return self.model.predict(x)

    def _get_y_minimum(self) -> np.ndarray:
        """Return the minimum value in the samples observed so far."""
        return np.min(self.model.Y, axis=0)

class MeanPluginExpectedImprovement(ExpectedImprovement):
    def __init__(self, model: IModelWithNoise, jitter: float = 0.0) -> None:
        """
        A variant of expected improvement that accounts for observation noise.

        For a given input, this acquisition computes the expected improvement over the *mean* at the
        best point observed so far.

        This is a heuristic that allows Expected Improvement to deal with problems with noisy observations, where
        the standard Expected Improvement might behave undesirably if the noise is too large.

        For more information see:
            "A benchmark of kriging-based infill criteria for noisy optimization" by Picheny et al.
        Note: the model type should be Union[IPredictsWithNoise, Intersection[IpredictsWithNoise, IDifferentiable]].
            Support for Intersection types might be added to Python in the future (see PEP 483)

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        super().__init__(model=model, jitter=jitter)

    def _get_y_minimum(self) -> np.ndarray:
        """Return the smallest model mean prediction at the previously observed points."""
        means_at_prev, _ = self.model.predict_noiseless(self.model.X)
        return np.min(means_at_prev, axis=0)

    def _get_model_predictions(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """Return the likelihood-free (i.e. without observation noise) prediction from the model."""
        return self.model.predict_noiseless(x)

