class TraceEnum_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI that supports
    - exhaustive enumeration over discrete sample sites, and
    - local parallel sampling over any sample site in the guide.

    To enumerate over a sample site in the ``guide``, mark the site with either
    ``infer={'enumerate': 'sequential'}`` or
    ``infer={'enumerate': 'parallel'}``. To configure all guide sites at once,
    use :func:`~pyro.infer.enum.config_enumerate`. To enumerate over a sample
    site in the ``model``, mark the site ``infer={'enumerate': 'parallel'}``
    and ensure the site does not appear in the ``guide``.

    This assumes restricted dependency structure on the model and guide:
    variables outside of an :class:`~pyro.plate` can never depend on
    variables inside that :class:`~pyro.plate`.
    """

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )

        if is_validation_enabled():
            check_traceenum_requirements(model_trace, guide_trace)
            _check_tmc_elbo_constraint(model_trace, guide_trace)

            has_enumerated_sites = any(
                site["infer"].get("enumerate")
                for trace in (guide_trace, model_trace)
                for name, site in trace.nodes.items()
                if site["type"] == "sample"
            )

            if self.strict_enumeration_warning and not has_enumerated_sites:
                warnings.warn(
                    "TraceEnum_ELBO found no sample sites configured for enumeration. "
                    "If you want to enumerate sites, you need to @config_enumerate or set "
                    'infer={"enumerate": "sequential"} or infer={"enumerate": "parallel"}? '
                    "If you do not want to enumerate, consider using Trace_ELBO instead."
                )

        guide_trace.pack_tensors()
        model_trace.pack_tensors(guide_trace.plate_to_symbol)
        return model_trace, guide_trace

    def _get_traces(self, model, guide, args, kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if isinstance(poutine.unwrap(guide), poutine.messenger.Messenger):
            raise NotImplementedError("TraceEnum_ELBO does not support GuideMessenger")
        if self.max_plate_nesting == float("inf"):
            self._guess_max_plate_nesting(model, guide, args, kwargs)
        if self.vectorize_particles:
            guide = self._vectorized_num_particles(guide)
            model = self._vectorized_num_particles(model)

        # Enable parallel enumeration over the vectorized guide and model.
        # The model allocates enumeration dimensions after (to the left of) the guide,
        # accomplished by preserving the _ENUM_ALLOCATOR state after the guide call.
        guide_enum = EnumMessenger(first_available_dim=-1 - self.max_plate_nesting)
        model_enum = EnumMessenger()  # preserve _ENUM_ALLOCATOR state
        guide = guide_enum(guide)
        model = model_enum(model)

        q = queue.LifoQueue()
        guide = poutine.queue(
            guide, q, escape_fn=iter_discrete_escape, extend_fn=iter_discrete_extend
        )
        for i in range(1 if self.vectorize_particles else self.num_particles):
            q.put(poutine.Trace())
            while not q.empty():
                yield self._get_trace(model, guide, args, kwargs)

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: an estimate of the ELBO
        :rtype: float

        Estimates the ELBO using ``num_particles`` many samples (particles).
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo += elbo_particle.item() / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        :returns: a differentiable estimate of the ELBO
        :rtype: torch.Tensor
        :raises ValueError: if the ELBO is not differentiable (e.g. is
            identically zero)

        Estimates a differentiable ELBO using ``num_particles`` many samples
        (particles).  The result should be infinitely differentiable (as long
        as underlying derivatives have been implemented).
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo = elbo + elbo_particle
        elbo = elbo / self.num_particles

        if not torch.is_tensor(elbo) or not elbo.requires_grad:
            raise ValueError("ELBO is cannot be differentiated: {}".format(elbo))

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: an estimate of the ELBO
        :rtype: float

        Estimates the ELBO using ``num_particles`` many samples (particles).
        Performs backward on the ELBO of each particle.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo += elbo_particle.item() / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = any(
                site["type"] == "param"
                for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
            )

            if trainable_params and elbo_particle.requires_grad:
                loss_particle = -elbo_particle
                (loss_particle / self.num_particles).backward(retain_graph=True)

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def compute_marginals(self, model, guide, *args, **kwargs):
        """
        Computes marginal distributions at each model-enumerated sample site.

        :returns: a dict mapping site name to marginal ``Distribution`` object
        :rtype: OrderedDict
        """
        if self.num_particles != 1:
            raise NotImplementedError(
                "TraceEnum_ELBO.compute_marginals() is not "
                "compatible with multiple particles."
            )
        model_trace, guide_trace = next(self._get_traces(model, guide, args, kwargs))
        for site in guide_trace.nodes.values():
            if site["type"] == "sample":
                if "_enumerate_dim" in site["infer"] or "_enum_total" in site["infer"]:
                    raise NotImplementedError(
                        "TraceEnum_ELBO.compute_marginals() is not "
                        "compatible with guide enumeration."
                    )
        return _compute_marginals(model_trace, guide_trace)

    def sample_posterior(self, model, guide, *args, **kwargs):
        """
        Sample from the joint posterior distribution of all model-enumerated sites given all observations
        """
        if self.num_particles != 1:
            raise NotImplementedError(
                "TraceEnum_ELBO.sample_posterior() is not "
                "compatible with multiple particles."
            )
        with poutine.block(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Found vars in model but not guide")
            model_trace, guide_trace = next(
                self._get_traces(model, guide, args, kwargs)
            )

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                if "_enumerate_dim" in site["infer"] or "_enum_total" in site["infer"]:
                    raise NotImplementedError(
                        "TraceEnum_ELBO.sample_posterior() is not "
                        "compatible with guide enumeration."
                    )

        # TODO replace BackwardSample with torch_sample backend to ubersum
        with BackwardSampleMessenger(model_trace, guide_trace):
            return poutine.replay(model, trace=guide_trace)(*args, **kwargs)

