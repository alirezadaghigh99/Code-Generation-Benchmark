class CoalescentTimes(TorchDistribution):
    """
    Distribution over sorted coalescent times given irregular sampled
    ``leaf_times`` and constant population size.

    Sample values will be **sorted** sets of binary coalescent times. Each
    sample ``value`` will have cardinality ``value.size(-1) =
    leaf_times.size(-1) - 1``, so that phylogenies are complete binary trees.
    This distribution can thus be batched over multiple samples of phylogenies
    given fixed (number of) leaf times, e.g. over phylogeny samples from BEAST
    or MrBayes.

    **References**

    [1] J.F.C. Kingman (1982)
        "On the Genealogy of Large Populations"
        Journal of Applied Probability
    [2] J.F.C. Kingman (1982)
        "The Coalescent"
        Stochastic Processes and their Applications

    :param torch.Tensor leaf_times: Vector of times of sampling events, i.e.
        leaf nodes in the phylogeny. These can be arbitrary real numbers with
        arbitrary order and duplicates.
    :param torch.Tensor rate: Base coalescent rate (pairwise rate of
        coalescence) under a constant population size model. Defaults to 1.
    """

    arg_constraints = {"leaf_times": constraints.real, "rate": constraints.positive}

    def __init__(self, leaf_times, rate=1.0, *, validate_args=None):
        rate = torch.as_tensor(rate, dtype=leaf_times.dtype, device=leaf_times.device)
        batch_shape = broadcast_shape(rate.shape, leaf_times.shape[:-1])
        event_shape = (leaf_times.size(-1) - 1,)
        self.leaf_times = leaf_times
        self.rate = rate
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return CoalescentTimesConstraint(self.leaf_times)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        coal_times = value
        phylogeny = _make_phylogeny(self.leaf_times, coal_times)

        # The coalescent process is like a Poisson process with rate binomial
        # in the number of lineages, which changes at each event.
        binomial = phylogeny.binomial[..., :-1]
        interval = phylogeny.times[..., :-1] - phylogeny.times[..., 1:]
        log_prob = self.rate.log() * coal_times.size(-1) - self.rate * (
            binomial * interval
        ).sum(-1)

        # Scaling by those rates and accounting for log|jacobian|, the density
        # is that of a collection of independent Exponential intervals.
        log_abs_det_jacobian = phylogeny.coal_binomial.log().sum(-1).neg()
        return log_prob - log_abs_det_jacobian

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)[:-1]
        leaf_times = self.leaf_times.expand(shape + (-1,))
        return _sample_coalescent_times(leaf_times)

