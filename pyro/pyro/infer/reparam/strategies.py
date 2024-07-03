class MinimalReparam(Strategy):
    """
    Minimal reparametrization strategy that reparametrizes only those sites
    that would otherwise lead to error, e.g.
    :class:`~pyro.distributions.Stable` and
    :class:`~pyro.distributions.ProjectedNormal` random variables.

    Example::

        @MinimalReparam()
        def model(...):
            ...

    which is equivalent to::

        @poutine.reparam(config=MinimalReparam())
        def model(...):
            ...
    """

    def configure(self, msg: dict) -> Optional[Reparam]:
        return _minimal_reparam(msg["fn"], msg["is_observed"])class AutoReparam(Strategy):
    """
    Applies a recommended set of reparametrizers. These currently include:
    :class:`MinimalReparam`,
    :class:`~pyro.infer.reparam.transform.TransformReparam`, a fully-learnable
    :class:`~pyro.infer.reparam.loc_scale.LocScaleReparam`, and
    :class:`~pyro.infer.reparam.softmax.GumbelSoftmaxReparam`.

    Example::

        @AutoReparam()
        def model(...):
            ...

    which is equivalent to::

        @poutine.reparam(config=AutoReparam())
        def model(...):
            ...

    .. warning:: This strategy may change behavior across Pyro releases.
        To inspect or save a given behavior, extract the ``.config`` dict after
        running the model at least once.

    :param centered: Optional centering parameter for
        :class:`~pyro.infer.reparam.loc_scale.LocScaleReparam` reparametrizers.
        If None (default), centering will be learned. If a float in
        ``[0.0,1.0]``, then a fixed centering. To completely decenter (e.g. in
        MCMC), set to 0.0.
    """

    def __init__(self, *, centered: Optional[float] = None):
        assert centered is None or isinstance(centered, float)
        super().__init__()
        self.centered = centered

    def configure(self, msg: dict) -> Optional[Reparam]:
        # Focus on tricks for latent sites.
        fn = msg["fn"]
        if not msg["is_observed"]:
            # Unwrap Independent, Masked, Transformed etc.
            while isinstance(getattr(fn, "base_dist", None), dist.Distribution):
                if isinstance(fn, torch.distributions.TransformedDistribution):
                    return TransformReparam()  # Then reparametrize new sites.
                fn = fn.base_dist

            # Try to apply a GumbelSoftmaxReparam.
            if isinstance(fn, torch.distributions.RelaxedOneHotCategorical):
                return GumbelSoftmaxReparam()

            # Apply a learnable LocScaleReparam.
            result = _loc_scale_reparam(msg["name"], fn, self.centered)
            if result is not None:
                return result

        # Apply minimal reparametrizers.
        return _minimal_reparam(fn, msg["is_observed"])