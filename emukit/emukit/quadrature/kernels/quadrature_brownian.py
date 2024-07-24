class QuadratureProductBrownianLebesgueMeasure(QuadratureProductBrownian, LebesgueEmbedding):
    """A product Brownian kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureProductBrownian`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param brownian_kernel: The standard EmuKit product Brownian kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, brownian_kernel: IProductBrownian, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(brownian_kernel=brownian_kernel, measure=measure, variable_names=variable_names)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.variance * z

    def _get_univariate_parameters(self, dim: int) -> dict:
        return {
            "domain": self.measure.domain.bounds[dim],
            "offset": self.offset,
            "normalize": self.measure.is_normalized,
        }

    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        offset = parameters["offset"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        kernel_mean = b * x - 0.5 * x**2 - 0.5 * a**2
        return (kernel_mean.T - offset * (b - a)) * normalization

    def _qKq_1d(self, **parameters) -> float:
        a, b = parameters["domain"]
        offset = parameters["offset"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        qKq = 0.5 * b * (b**2 - a**2) - (b**3 - a**3) / 6 - 0.5 * a**2 * (b - a)
        return (float(qKq) - offset * (b - a) ** 2) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        return (b - x).T * normalization

class QuadratureBrownianLebesgueMeasure(QuadratureBrownian, LebesgueEmbedding):
    """A Brownian motion kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IBrownian`
       * :class:`emukit.quadrature.kernels.QuadratureBrownian`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param brownian_kernel: The standard EmuKit Brownian motion kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, brownian_kernel: IBrownian, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(brownian_kernel=brownian_kernel, measure=measure, variable_names=variable_names)

    def qK(self, x2: np.ndarray) -> np.ndarray:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        kernel_mean = ub * x2 - 0.5 * x2**2 - 0.5 * lb**2
        return (self.variance * self.measure.density) * kernel_mean.T

    def qKq(self) -> float:
        lb = self.measure.domain.lower_bounds[None, :]
        ub = self.measure.domain.upper_bounds[None, :]
        qKq = 0.5 * ub * (ub**2 - lb**2) - (ub**3 - lb**3) / 6 - 0.5 * lb**2 * (ub - lb)
        return (self.measure.density**2 * self.variance) * float(qKq)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        ub = self.measure.domain.upper_bounds[None, :]
        return (self.measure.density * self.variance) * (ub - x2).T

