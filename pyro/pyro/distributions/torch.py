class Normal(torch.distributions.Normal, TorchDistributionMixin):
    passclass Categorical(torch.distributions.Categorical, TorchDistributionMixin):
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
        return resultclass Beta(torch.distributions.Beta, TorchDistributionMixin):
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
        return updated, log_normalizerclass Normal(torch.distributions.Normal, TorchDistributionMixin):
    pass