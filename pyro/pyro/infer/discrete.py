class TraceEnumSample_ELBO(TraceEnum_ELBO):
    """
    This extends :class:`TraceEnum_ELBO` to make it cheaper to sample from
    discrete latent states during SVI.

    The following are equivalent but the first is cheaper, sharing work
    between the computations of ``loss`` and ``z``::

        # Version 1.
        elbo = TraceEnumSample_ELBO(max_plate_nesting=1)
        loss = elbo.loss(*args, **kwargs)
        z = elbo.sample_saved()

        # Version 2.
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        loss = elbo.loss(*args, **kwargs)
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        z = infer_discrete(poutine.replay(model, guide_trace),
                           first_available_dim=-2)(*args, **kwargs)

    """

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        # Mark all sample sites with require_backward to gather enumerated
        # sites and adjust cond_indep_stack of all sample sites.
        for node in model_trace.nodes.values():
            if node["type"] == "sample" and not node["is_observed"]:
                log_prob = node["packed"]["unscaled_log_prob"]
                require_backward(log_prob)

        self._saved_state = model, model_trace, guide_trace, args, kwargs
        return model_trace, guide_trace

    def sample_saved(self):
        """
        Generate latent samples while reusing work from SVI.step().
        """
        model, model_trace, guide_trace, args, kwargs = self._saved_state
        model = poutine.replay(model, guide_trace)
        temperature = 1
        return _sample_posterior_from_trace(
            model,
            model_trace,
            temperature,
            self.strict_enumeration_warning,
            *args,
            **kwargs
        )

