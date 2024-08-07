class AutoNormal(AutoGuide):
    """This implementation of :class:`AutoGuide` uses a Normal distribution
    with a diagonal covariance matrix to construct a guide over the entire
    latent space. The guide does not depend on the model's ``*args, **kwargs``.

    It should be equivalent to :class: `AutoDiagonalNormal` , but with
    more convenient site names and with better support for
    :class:`~pyro.infer.trace_mean_field_elbo.TraceMeanField_ELBO` .

    In :class:`AutoDiagonalNormal` , if your model has N named
    parameters with dimensions k_i and sum k_i = D, you get a single
    vector of length D for your mean, and a single vector of length D
    for sigmas.  This guide gives you N distinct normals that you can
    call by name.

    Usage::

        guide = AutoNormal(model)
        svi = SVI(model, guide, ...)

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    :param callable create_plates: An optional function inputing the same
        ``*args,**kwargs`` as ``model()`` and returning a :class:`pyro.plate`
        or iterable of plates. Plates not returned will be created
        automatically as usual. This is useful for data subsampling.
    """

    scale_constraint = constraints.softplus_positive

    def __init__(
        self, model, *, init_loc_fn=init_to_feasible, init_scale=0.1, create_plates=None
    ):
        self.init_loc_fn = init_loc_fn

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale

        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._event_dims = {}
        self.locs = PyroModule()
        self.scales = PyroModule()

        # Initialize guide params
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect unconstrained event_dims, which may differ from constrained event_dims.
            with helpful_support_errors(site):
                init_loc = (
                    biject_to(site["fn"].support).inv(site["value"].detach()).detach()
                )
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = frame.full_size or frame.size
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = periodic_repeat(init_loc, full_size, dim).contiguous()
            init_scale = torch.full_like(init_loc, self._init_scale)

            deep_setattr(
                self.locs, name, PyroParam(init_loc, constraints.real, event_dim)
            )
            deep_setattr(
                self.scales,
                name,
                PyroParam(init_scale, self.scale_constraint, event_dim),
            )

    def _get_loc_and_scale(self, name):
        site_loc = deep_getattr(self.locs, name)
        site_scale = deep_getattr(self.scales, name)
        return site_loc, site_scale

    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        .. note:: This method is used internally by :class:`~torch.nn.Module`.
            Users should instead use :meth:`~torch.nn.Module.__call__`.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)
                unconstrained_latent = pyro.sample(
                    name + "_unconstrained",
                    dist.Normal(
                        site_loc,
                        site_scale,
                    ).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )

                value = transform(unconstrained_latent)
                if poutine.get_mask() is False:
                    log_density = 0.0
                else:
                    log_density = transform.inv.log_abs_det_jacobian(
                        value,
                        unconstrained_latent,
                    )
                    log_density = sum_rightmost(
                        log_density,
                        log_density.dim() - value.dim() + site["fn"].event_dim,
                    )
                delta_dist = dist.Delta(
                    value,
                    log_density=log_density,
                    event_dim=site["fn"].event_dim,
                )

                result[name] = pyro.sample(name, delta_dist)

        return result

    @torch.no_grad()
    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.

        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        medians = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, _ = self._get_loc_and_scale(name)
            median = biject_to(site["fn"].support)(site_loc)
            if median is site_loc:
                median = median.clone()
            medians[name] = median

        return medians

    @torch.no_grad()
    def quantiles(self, quantiles, *args, **kwargs):
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a tensor of quantile values.
        :rtype: dict
        """
        results = {}

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, site_scale = self._get_loc_and_scale(name)

            site_quantiles = torch.tensor(
                quantiles, dtype=site_loc.dtype, device=site_loc.device
            )
            site_quantiles = site_quantiles.reshape((-1,) + (1,) * site_loc.dim())
            site_quantiles_values = dist.Normal(site_loc, site_scale).icdf(
                site_quantiles
            )
            constrained_site_quantiles = biject_to(site["fn"].support)(
                site_quantiles_values
            )
            results[name] = constrained_site_quantiles

        return results

