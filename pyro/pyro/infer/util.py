class MultiFrameTensor(dict):
    """
    A container for sums of Tensors among different :class:`plate` contexts.

    Used in :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO` to simplify
    downstream cost computation logic.

    Example::

        downstream_cost = MultiFrameTensor()
        for site in downstream_nodes:
            downstream_cost.add((site["cond_indep_stack"], site["log_prob"]))
        downstream_cost.add(*other_costs.items())  # add in bulk
        summed = downstream_cost.sum_to(target_site["cond_indep_stack"])
    """

    def __init__(self, *items):
        super().__init__()
        self.add(*items)

    def add(self, *items):
        """
        Add a collection of (cond_indep_stack, tensor) pairs. Keys are
        ``cond_indep_stack``s, i.e. tuples of :class:`CondIndepStackFrame`s.
        Values are :class:`torch.Tensor`s.
        """
        for cond_indep_stack, value in items:
            frames = frozenset(f for f in cond_indep_stack if f.vectorized)
            assert all(f.dim < 0 and -value.dim() <= f.dim for f in frames)
            if frames in self:
                self[frames] = self[frames] + value
            else:
                self[frames] = value

    def sum_to(self, target_frames):
        total = None
        for frames, value in self.items():
            for f in frames:
                if f not in target_frames and value.shape[f.dim] != 1:
                    value = value.sum(f.dim, True)
            while value.shape and value.shape[0] == 1:
                value = value.squeeze(0)
            total = value if total is None else total + value
        return 0.0 if total is None else total

    def __repr__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ",\n\t".join(["({}, ...)".format(frames) for frames in self]),
        )

