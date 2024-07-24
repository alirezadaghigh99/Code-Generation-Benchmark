class RejectionStandardGamma(Rejector):
    """
    Naive Marsaglia & Tsang rejection sampler for standard Gamma distibution.
    This assumes `concentration >= 1` and does not boost `concentration` or augment shape.
    """

    def __init__(self, concentration):
        if concentration.data.min() < 1:
            raise NotImplementedError("concentration < 1 is not supported")
        self.concentration = concentration
        self._standard_gamma = Gamma(
            concentration, concentration.new([1.0]).squeeze().expand_as(concentration)
        )
        # The following are Marsaglia & Tsang's variable names.
        self._d = self.concentration - 1.0 / 3.0
        self._c = 1.0 / torch.sqrt(9.0 * self._d)
        # Compute log scale using Gamma.log_prob().
        x = self._d.detach()  # just an arbitrary x.
        log_scale = (
            self.propose_log_prob(x) + self.log_prob_accept(x) - self.log_prob(x)
        )
        super().__init__(
            self.propose,
            self.log_prob_accept,
            log_scale,
            batch_shape=concentration.shape,
            event_shape=(),
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RejectionStandardGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new._standard_gamma = self._standard_gamma.expand(batch_shape)
        new._d = self._d.expand(batch_shape)
        new._c = self._c.expand(batch_shape)
        # Compute log scale using Gamma.log_prob().
        x = new._d.detach()  # just an arbitrary x.
        log_scale = new.propose_log_prob(x) + new.log_prob_accept(x) - new.log_prob(x)
        super(RejectionStandardGamma, new).__init__(
            new.propose,
            new.log_prob_accept,
            log_scale,
            batch_shape=batch_shape,
            event_shape=(),
        )
        new._validate_args = self._validate_args
        return new

    @weakmethod
    def propose(self, sample_shape=torch.Size()):
        # Marsaglia & Tsang's x == Naesseth's epsilon`
        x = torch.randn(
            sample_shape + self.concentration.shape,
            dtype=self.concentration.dtype,
            device=self.concentration.device,
        )
        y = 1.0 + self._c * x
        v = y * y * y
        return (self._d * v).clamp_(1e-30, 1e30)

    def propose_log_prob(self, value):
        v = value / self._d
        result = -self._d.log()
        y = v.pow(1 / 3)
        result -= torch.log(3 * y**2)
        x = (y - 1) / self._c
        result -= self._c.log()
        result += Normal(
            torch.zeros_like(self.concentration), torch.ones_like(self.concentration)
        ).log_prob(x)
        return result

    @weakmethod
    def log_prob_accept(self, value):
        v = value / self._d
        y = torch.pow(v, 1.0 / 3.0)
        x = (y - 1.0) / self._c
        log_prob_accept = 0.5 * x * x + self._d * (1.0 - v + torch.log(v))
        log_prob_accept[y <= 0] = -float("inf")
        return log_prob_accept

    def log_prob(self, x):
        return self._standard_gamma.log_prob(x)

