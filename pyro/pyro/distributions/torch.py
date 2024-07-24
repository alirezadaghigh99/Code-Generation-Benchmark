class Normal(torch.distributions.Normal, TorchDistributionMixin):
    pass

class Categorical(torch.distributions.Categorical, TorchDistributionMixin):
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}

    def log_prob(self, value):
        if getattr(value, "_pyro_categorical_support", None) == id(self):
            # Assume value is a reshaped torch.arange(event_shape[0]).
            # In this case we can call .reshape() rather than torch.gather().
            if not torch._C._get_tracing_state():
                if self._validate_args:
                    self._validate_sample(value)
                assert value.size(0) == self.logits.size(-1)
            logits = self.logits
            if logits.dim() <= value.dim():
                logits = logits.reshape(
                    (1,) * (1 + value.dim() - logits.dim()) + logits.shape
                )
            if not torch._C._get_tracing_state():
                assert logits.size(-1 - value.dim()) == 1
            return logits.transpose(-1 - value.dim(), -1).squeeze(-1)
        return super().log_prob(value)

    def enumerate_support(self, expand=True):
        result = super().enumerate_support(expand=expand)
        if not expand:
            result._pyro_categorical_support = id(self)
        return result

class Beta(torch.distributions.Beta, TorchDistributionMixin):
    def conjugate_update(self, other):
        """
        EXPERIMENTAL.
        """
        assert isinstance(other, Beta)
        concentration1 = self.concentration1 + other.concentration1 - 1
        concentration0 = self.concentration0 + other.concentration0 - 1
        updated = Beta(concentration1, concentration0)

        def _log_normalizer(d):
            x = d.concentration1
            y = d.concentration0
            return (x + y).lgamma() - x.lgamma() - y.lgamma()

        log_normalizer = (
            _log_normalizer(self) + _log_normalizer(other) - _log_normalizer(updated)
        )
        return updated, log_normalizer

class Uniform(torch.distributions.Uniform, TorchDistributionMixin):
    def __init__(self, low, high, validate_args=None):
        self._unbroadcasted_low = low
        self._unbroadcasted_high = high
        super().__init__(low, high, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Uniform, _instance)
        new = super().expand(batch_shape, _instance=new)
        new._unbroadcasted_low = self._unbroadcasted_low
        new._unbroadcasted_high = self._unbroadcasted_high
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self._unbroadcasted_low, self._unbroadcasted_high)

class MultivariateNormal(
    torch.distributions.MultivariateNormal, TorchDistributionMixin
):
    @staticmethod
    def infer_shapes(
        loc, covariance_matrix=None, precision_matrix=None, scale_tril=None
    ):
        batch_shape, event_shape = loc[:-1], loc[-1:]
        for matrix in [covariance_matrix, precision_matrix, scale_tril]:
            if matrix is not None:
                batch_shape = broadcast_shape(batch_shape, matrix[:-2])
        return batch_shape, event_shape

class Dirichlet(torch.distributions.Dirichlet, TorchDistributionMixin):
    @staticmethod
    def infer_shapes(concentration):
        batch_shape = concentration[:-1]
        event_shape = concentration[-1:]
        return batch_shape, event_shape

    def conjugate_update(self, other):
        """
        EXPERIMENTAL.
        """
        assert isinstance(other, Dirichlet)
        concentration = self.concentration + other.concentration - 1
        updated = Dirichlet(concentration)

        def _log_normalizer(d):
            c = d.concentration
            return c.sum(-1).lgamma() - c.lgamma().sum(-1)

        log_normalizer = (
            _log_normalizer(self) + _log_normalizer(other) - _log_normalizer(updated)
        )
        return updated, log_normalizer

class OneHotCategorical(torch.distributions.OneHotCategorical, TorchDistributionMixin):
    @staticmethod
    def infer_shapes(probs=None, logits=None):
        tensor = probs if logits is None else logits
        event_shape = tensor[-1:]
        batch_shape = tensor[:-1]
        return batch_shape, event_shape

class Binomial(torch.distributions.Binomial, TorchDistributionMixin):
    # EXPERIMENTAL threshold on total_count above which sampling will use a
    # clamped Poisson approximation for Binomial samples. This is useful for
    # sampling very large populations.
    approx_sample_thresh = math.inf

    # EXPERIMENTAL If set to a positive value, the .log_prob() method will use
    # a shifted Sterling's approximation to the Beta function, reducing
    # computational cost from 3 lgamma() evaluations to 4 log() evaluations
    # plus arithmetic. Recommended values are between 0.1 and 0.01.
    approx_log_prob_tol = 0.0

    def sample(self, sample_shape=torch.Size()):
        if self.approx_sample_thresh < math.inf:
            exact = self.total_count <= self.approx_sample_thresh
            if not exact.all():
                # Approximate large counts with a moment-matched clamped Poisson.
                with torch.no_grad():
                    shape = self._extended_shape(sample_shape)
                    p = self.probs
                    q = 1 - self.probs
                    mean = torch.min(p, q) * self.total_count
                    variance = p * q * self.total_count
                    shift = (mean - variance).round()
                    result = torch.poisson(variance.expand(shape))
                    result = torch.min(result + shift, self.total_count)
                    sample = torch.where(p < q, result, self.total_count - result)
                # Draw exact samples for remaining items.
                if exact.any():
                    total_count = torch.where(
                        exact, self.total_count, torch.zeros_like(self.total_count)
                    )
                    exact_sample = torch.distributions.Binomial(
                        total_count, self.probs, validate_args=False
                    ).sample(sample_shape)
                    sample = torch.where(exact, exact_sample, sample)
                return sample
        return super().sample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        n = self.total_count
        k = value
        # k * log(p) + (n - k) * log(1 - p) = k * (log(p) - log(1 - p)) + n * log(1 - p)
        #     (case logit < 0)              = k * logit - n * log1p(e^logit)
        #     (case logit > 0)              = k * logit - n * (log(p) - log(1 - p)) + n * log(p)
        #                                   = k * logit - n * logit - n * log1p(e^-logit)
        #     (merge two cases)             = k * logit - n * max(logit, 0) - n * log1p(e^-|logit|)
        normalize_term = n * (
            _clamp_by_zero(self.logits) + self.logits.abs().neg().exp().log1p()
        )
        return (
            k * self.logits
            - normalize_term
            + log_binomial(n, k, tol=self.approx_log_prob_tol)
        )

