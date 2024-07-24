class QuadratureRBFLebesgueMeasure(QuadratureRBF, LebesgueEmbedding):
    """An RBF kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.kernels.QuadratureRBF`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param rbf_kernel: The standard EmuKit rbf-kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, rbf_kernel: IRBF, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(rbf_kernel=rbf_kernel, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray) -> np.ndarray:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        erf_lo = erf(self._scaled_vector_diff(lb, x2))
        erf_up = erf(self._scaled_vector_diff(ub, x2))
        kernel_mean = (np.sqrt(np.pi / 2.0) * self.lengthscales * (erf_up - erf_lo)).prod(axis=1)
        return (self.variance * self.measure.density) * kernel_mean.reshape(1, -1)

    def qKq(self) -> float:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        diff_bounds_scaled = self._scaled_vector_diff(ub, lb)
        exp_term = (np.exp(-(diff_bounds_scaled**2)) - 1.0) / np.sqrt(np.pi)
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled
        qKq = ((2 * np.sqrt(np.pi) * self.lengthscales**2) * (exp_term + erf_term)).prod()
        return (self.variance * self.measure.density**2) * float(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        exp_lo = np.exp(-self._scaled_vector_diff(x2, lb) ** 2)
        exp_up = np.exp(-self._scaled_vector_diff(x2, ub) ** 2)
        erf_lo = erf(self._scaled_vector_diff(lb, x2))
        erf_up = erf(self._scaled_vector_diff(ub, x2))
        fraction = ((exp_lo - exp_up) / (self.lengthscales * np.sqrt(np.pi / 2.0) * (erf_up - erf_lo))).T
        return self.qK(x2) * fraction

class QuadratureRBFGaussianMeasure(QuadratureRBF, GaussianEmbedding):
    """An RBF kernel augmented with integrability w.r.t. a Gaussian measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.kernels.QuadratureRBF`
       * :class:`emukit.quadrature.measures.GaussianMeasure`

    :param rbf_kernel: The standard EmuKit rbf-kernel.
    :param measure: A Gaussian measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, rbf_kernel: IRBF, measure: GaussianMeasure, variable_names: str = "") -> None:
        super().__init__(rbf_kernel=rbf_kernel, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        lengthscales = scale_factor * self.lengthscales
        sigma2 = self.measure.variance
        mu = self.measure.mean
        factor = np.sqrt(lengthscales**2 / (lengthscales**2 + sigma2)).prod()
        scaled_norm_sq = np.power(self._scaled_vector_diff(x2, mu, np.sqrt(lengthscales**2 + sigma2)), 2).sum(axis=1)
        return (self.variance * factor) * np.exp(-scaled_norm_sq).reshape(1, -1)

    def qKq(self) -> float:
        lengthscales = self.lengthscales
        qKq = np.sqrt(lengthscales**2 / (lengthscales**2 + 2 * self.measure.variance)).prod()
        return self.variance * float(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        scaled_diff = (x2 - self.measure.mean) / (self.lengthscales**2 + self.measure.variance)
        return -self.qK(x2) * scaled_diff.T

