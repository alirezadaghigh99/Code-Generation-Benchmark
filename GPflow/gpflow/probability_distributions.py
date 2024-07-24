class Gaussian(ProbabilityDistribution):
    @check_shapes(
        "mu: [N, D]",
        "cov: [N, D, D]",
    )
    def __init__(self, mu: TensorType, cov: TensorType):
        self.mu = mu
        self.cov = cov

    @property
    def shape(self) -> Shape:
        return self.mu.shape  # type: ignore[no-any-return]

class MarkovGaussian(ProbabilityDistribution):
    """
    Gaussian distribution with Markov structure.
    Only covariances and covariances between t and t+1 need to be
    parameterised. We use the solution proposed by Carl Rasmussen, i.e. to
    represent
    Var[x_t] = cov[x_t, :, :] * cov[x_t, :, :].T
    Cov[x_t, x_{t+1}] = cov[t, :, :] * cov[t+1, :, :]
    """

    @check_shapes(
        "mu: [N_plus_1, D]",
        "cov: [2, N_plus_1, D, D]",
    )
    def __init__(self, mu: TensorType, cov: TensorType):
        self.mu = mu
        self.cov = cov

    @property
    def shape(self) -> Shape:
        shape = self.mu.shape
        if shape is None:
            return shape
        N_plus_1, D = shape
        N = N_plus_1 - 1
        return N, D

