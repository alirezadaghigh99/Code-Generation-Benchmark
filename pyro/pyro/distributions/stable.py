class Stable(TorchDistribution):
    r"""
    Levy :math:`\alpha`-stable distribution. See [1] for a review.

    This uses Nolan's parametrization [2] of the ``loc`` parameter, which is
    required for continuity and differentiability. This corresponds to the
    notation :math:`S^0_\alpha(\beta,\sigma,\mu_0)` of [1], where
    :math:`\alpha` = stability, :math:`\beta` = skew, :math:`\sigma` = scale,
    and :math:`\mu_0` = loc. To instead use the S parameterization as in scipy,
    pass ``coords="S"``, but BEWARE this is discontinuous at ``stability=1``
    and has poor geometry for inference.

    This implements a reparametrized sampler :meth:`rsample` , and a relatively
    expensive :meth:`log_prob` calculation by numerical integration which makes
    inference slow (compared to other distributions) , but with better
    convergence properties especially for :math:`\alpha`-stable distributions
    that are skewed (see the ``skew`` parameter below). Faster
    inference can be performed using either likelihood-free algorithms such as
    :class:`~pyro.infer.energy_distance.EnergyDistance`, or reparameterization
    via the :func:`~pyro.poutine.handlers.reparam` handler with one of the
    reparameterizers :class:`~pyro.infer.reparam.stable.LatentStableReparam` ,
    :class:`~pyro.infer.reparam.stable.SymmetricStableReparam` , or
    :class:`~pyro.infer.reparam.stable.StableReparam` e.g.::

        with poutine.reparam(config={"x": StableReparam()}):
            pyro.sample("x", Stable(stability, skew, scale, loc))

    or simply wrap in :class:`~pyro.infer.reparam.strategies.MinimalReparam` or
    :class:`~pyro.infer.reparam.strategies.AutoReparam` , e.g.::

        @MinimalReparam()
        def model():
            ...

    [1] S. Borak, W. Hardle, R. Weron (2005).
        Stable distributions.
        https://edoc.hu-berlin.de/bitstream/handle/18452/4526/8.pdf
    [2] J.P. Nolan (1997).
        Numerical calculation of stable densities and distribution functions.
    [3] Rafal Weron (1996).
        On the Chambers-Mallows-Stuck Method for
        Simulating Skewed Stable Random Variables.
    [4] J.P. Nolan (2017).
        Stable Distributions: Models for Heavy Tailed Data.
        https://edspace.american.edu/jpnolan/wp-content/uploads/sites/1720/2020/09/Chap1.pdf

    :param Tensor stability: Levy stability parameter :math:`\alpha\in(0,2]` .
    :param Tensor skew: Skewness :math:`\beta\in[-1,1]` .
    :param Tensor scale: Scale :math:`\sigma > 0` . Defaults to 1.
    :param Tensor loc: Location :math:`\mu_0` when using Nolan's S0
        parametrization [2], or :math:`\mu` when using the S parameterization.
        Defaults to 0.
    :param str coords: Either "S0" (default) to use Nolan's continuous S0
        parametrization, or "S" to use the discontinuous parameterization.
    """

    has_rsample = True
    arg_constraints = {
        "stability": constraints.interval(0, 2),  # half-open (0, 2]
        "skew": constraints.interval(-1, 1),  # closed [-1, 1]
        "scale": constraints.positive,
        "loc": constraints.real,
    }
    support = constraints.real

    def __init__(
        self, stability, skew, scale=1.0, loc=0.0, coords="S0", validate_args=None
    ):
        assert coords in ("S", "S0"), coords
        self.stability, self.skew, self.scale, self.loc = broadcast_all(
            stability, skew, scale, loc
        )
        self.coords = coords
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Stable, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        new.coords = self.coords
        super(Stable, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        r"""Implemented by numerical integration that is based on the algorithm
        proposed by Chambers, Mallows and Stuck (CMS) for simulating the
        Levy :math:`\alpha`-stable distribution. The CMS algorithm involves a
        nonlinear transformation of two independent random variables into
        one stable random variable. The first random variable is uniformly
        distributed while the second is exponentially distributed. The numerical
        integration is performed over the first uniformly distributed random
        variable.
        """
        if self._validate_args:
            self._validate_sample(value)

        # Undo shift and scale
        value = (value - self.loc) / self.scale
        value_dtype = value.dtype

        # Use double precision math
        alpha = self.stability.double()
        beta = self.skew.double()
        value = value.double()

        alpha, beta, value = broadcast_all(alpha, beta, value)

        log_prob = _stable_log_prob(alpha, beta, value, self.coords)

        return log_prob.to(dtype=value_dtype) - self.scale.log()

    def rsample(self, sample_shape=torch.Size()):
        # Draw parameter-free noise.
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            new_empty = self.stability.new_empty
            aux_uniform = new_empty(shape).uniform_(-math.pi / 2, math.pi / 2)
            aux_exponential = new_empty(shape).exponential_()

        # Differentiably transform.
        x = _standard_stable(
            self.stability, self.skew, aux_uniform, aux_exponential, coords=self.coords
        )
        return self.loc + self.scale * x

    @property
    def mean(self):
        result = self.loc
        if self.coords == "S0":
            result = (
                result - self.scale * self.skew * (math.pi / 2 * self.stability).tan()
            )
        return result.masked_fill(self.stability <= 1, math.nan)

    @property
    def variance(self):
        var = self.scale * self.scale
        return var.mul(2).masked_fill(self.stability < 2, math.inf)