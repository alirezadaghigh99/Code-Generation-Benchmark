class MaxValueEntropySearch(Acquisition):
    def __init__(
        self,
        model: Union[IModel, IEntropySearchModel],
        space: ParameterSpace,
        num_samples: int = 10,
        grid_size: int = 5000,
    ) -> None:
        """
        MES acquisition function approximates the distribution of the value at the global
        minimum and tries to decrease its entropy. See this paper for more details:
        Z. Wang, S. Jegelka
        Max-value Entropy Search for Efficient Bayesian Optimization
        ICML 2017

        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param num_samples: integer determining how many samples to draw of the minimum (does not need to be large)
        :param grid_size: number of random locations in grid used to fit the gumbel distribution and approximately generate
        the samples of the minimum (recommend scaling with problem dimension, i.e. 10000*d)
        """
        super().__init__()

        if not isinstance(model, IEntropySearchModel):
            raise RuntimeError("Model is not supported for MES")

        self.model = model
        self.space = space
        self.num_samples = num_samples
        self.grid_size = grid_size

        # Initialize parameters to lazily compute them once needed
        self.mins = None

    def update_parameters(self):
        """
        MES requires acces to a sample of possible minimum values y* of the objective function.
        To build this sample we approximate the empirical c.d.f of Pr(y*<y) with a Gumbel(a,b) distribution.
        This Gumbel distribution can then be easily sampled to yield approximate samples of y*

        This needs to be called once at the start of each BO step.
        """

        # First we generate a random grid of locations at which to fit the Gumbel distribution
        random_design = RandomDesign(self.space)
        grid = random_design.get_samples(self.grid_size)
        # also add the locations already queried in the previous BO steps
        grid = np.vstack([self.model.X, grid])
        # Get GP posterior at these points
        fmean, fvar = self.model.predict(grid)
        fsd = np.sqrt(fvar)

        # fit Gumbel distriubtion
        a, b = _fit_gumbel(fmean, fsd)

        # sample K times from this Gumbel distribution using the inverse probability integral transform,
        # i.e. given a sample r ~ Unif[0,1] then g = a + b * log( -1 * log(1 - r)) follows g ~ Gumbel(a,b).

        uniform_samples = np.random.rand(self.num_samples)
        gumbel_samples = np.log(-1 * np.log(1 - uniform_samples)) * b + a
        self.mins = gumbel_samples

    def _required_parameters_initialized(self):
        """
        Checks if all required parameters are initialized.
        """
        return self.mins is not None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the predicted change in entropy of p_min (the distribution
        of the minimal value of the objective function) if we evaluate x.
        :param x: points where the acquisition is evaluated.
        """
        if not self._required_parameters_initialized():
            self.update_parameters()

        # Calculate GP posterior at candidate points
        fmean, fvar = self.model.predict(x)
        fsd = np.sqrt(fvar)
        # Clip below to improve numerical stability
        fsd = np.maximum(fsd, 1e-10)

        # standardise
        gamma = (self.mins - fmean) / fsd

        minus_cdf = 1 - norm.cdf(gamma)
        # Clip  to improve numerical stability
        minus_cdf = np.clip(minus_cdf, a_min=1e-10, a_max=1)

        # calculate monte-carlo estimate of information gain
        f_acqu_x = np.mean(-gamma * norm.pdf(gamma) / (2 * minus_cdf) - np.log(minus_cdf), axis=1)
        return f_acqu_x.reshape(-1, 1)

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False

