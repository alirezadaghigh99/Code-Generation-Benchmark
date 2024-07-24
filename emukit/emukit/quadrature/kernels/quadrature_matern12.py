class QuadratureProductMatern12LebesgueMeasure(QuadratureProductMatern12, LebesgueEmbedding):
    """A product Matern12 kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern12`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern12`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param matern_kernel: The standard EmuKit product Matern12 kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, matern_kernel: IProductMatern12, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(matern_kernel=matern_kernel, measure=measure, variable_names=variable_names)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.variance * z

    def _get_univariate_parameters(self, dim: int) -> dict:
        return {
            "domain": self.measure.domain.bounds[dim],
            "lengthscale": self.lengthscales[dim],
            "normalize": self.measure.is_normalized,
        }

    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        first_term = -np.exp((a - x) / lengthscale)
        second_term = -np.exp((x - b) / lengthscale)
        return normalization * lengthscale * (2.0 + first_term + second_term)

    def _qKq_1d(self, **parameters) -> float:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        qKq = 2.0 * lengthscale * ((b - a) + lengthscale * (np.exp(-(b - a) / lengthscale) - 1.0))
        return float(qKq) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        first_term = np.exp((a - x) / lengthscale)
        second_term = -np.exp((x - b) / lengthscale)
        return (first_term + second_term) * normalization

