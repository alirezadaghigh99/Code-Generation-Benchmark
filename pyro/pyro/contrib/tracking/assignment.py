class MarginalAssignmentPersistent:
    """
    This computes marginal distributions of a multi-frame multi-object
    data association problem with an unknown number of persistent objects.

    The inputs are factors in a factor graph (existence probabilites for each
    potential object and assignment probabilities for each object-detection
    pair), and the outputs are marginal distributions of posterior existence
    probability of each potential object and posterior assignment probabilites
    of each object-detection pair.

    This assumes a shared (maximum) number of detections per frame; to handle
    variable number of detections, simply set corresponding elements of
    ``assign_logits`` to ``-float('inf')``.

    :param torch.Tensor exists_logits: a tensor of shape ``[num_objects]``
        representing per-object factors for existence of each potential object.
    :param torch.Tensor assign_logits: a tensor of shape
        ``[num_frames, num_detections, num_objects]`` representing per-edge
        factors of assignment probability, where each edge denotes that at a
        given time frame a given detection associates with a single object.
    :param int bp_iters: optional number of belief propagation iterations. If
        unspecified or ``None`` an expensive exact algorithm will be used.
    :param float bp_momentum: optional momentum to use for belief propagation.
        Should be in the interval ``[0,1)``.

    :ivar int num_frames: the number of time frames
    :ivar int num_detections: the (maximum) number of detections per frame
    :ivar int num_objects: the number of (potentially existing) objects
    :ivar pyro.distributions.Bernoulli exists_dist: a mean field posterior
        distribution over object existence.
    :ivar pyro.distributions.Categorical assign_dist: a mean field posterior
        distribution over the object (or None) to which each detection
        associates.  This has ``.event_shape == (num_objects + 1,)`` where the
        final element denotes spurious detection, and
        ``.batch_shape == (num_frames, num_detections)``.
    """

    def __init__(self, exists_logits, assign_logits, bp_iters=None, bp_momentum=0.5):
        assert exists_logits.dim() == 1, exists_logits.shape
        assert assign_logits.dim() == 3, assign_logits.shape
        assert assign_logits.shape[-1] == exists_logits.shape[-1]
        self.num_frames, self.num_detections, self.num_objects = assign_logits.shape

        # Clamp to avoid NANs.
        exists_logits = exists_logits.clamp(min=-40, max=40)
        assign_logits = assign_logits.clamp(min=-40, max=40)

        # This does all the work.
        if bp_iters is None:
            exists, assign = compute_marginals_persistent(exists_logits, assign_logits)
        else:
            exists, assign = compute_marginals_persistent_bp(
                exists_logits, assign_logits, bp_iters, bp_momentum
            )

        # Wrap the results in Distribution objects.
        # This adds a final logit=0 element denoting spurious detection.
        padded_assign = torch.nn.functional.pad(assign, (0, 1), "constant", 0.0)
        self.assign_dist = dist.Categorical(logits=padded_assign)
        self.exists_dist = dist.Bernoulli(logits=exists)
        assert self.assign_dist.batch_shape == (self.num_frames, self.num_detections)
        assert self.exists_dist.batch_shape == (self.num_objects,)

class MarginalAssignment:
    """
    Computes marginal data associations between objects and detections.

    This assumes that each detection corresponds to zero or one object,
    and each object corresponds to zero or more detections. Specifically
    this does not assume detections have been partitioned into frames of
    mutual exclusion as is common in 2-D assignment problems.

    :param torch.Tensor exists_logits: a tensor of shape ``[num_objects]``
        representing per-object factors for existence of each potential object.
    :param torch.Tensor assign_logits: a tensor of shape
        ``[num_detections, num_objects]`` representing per-edge factors of
        assignment probability, where each edge denotes that a given detection
        associates with a single object.
    :param int bp_iters: optional number of belief propagation iterations. If
        unspecified or ``None`` an expensive exact algorithm will be used.

    :ivar int num_detections: the number of detections
    :ivar int num_objects: the number of (potentially existing) objects
    :ivar pyro.distributions.Bernoulli exists_dist: a mean field posterior
        distribution over object existence.
    :ivar pyro.distributions.Categorical assign_dist: a mean field posterior
        distribution over the object (or None) to which each detection
        associates.  This has ``.event_shape == (num_objects + 1,)`` where the
        final element denotes spurious detection, and
        ``.batch_shape == (num_frames, num_detections)``.
    """

    def __init__(self, exists_logits, assign_logits, bp_iters=None):
        assert exists_logits.dim() == 1, exists_logits.shape
        assert assign_logits.dim() == 2, assign_logits.shape
        assert assign_logits.shape[-1] == exists_logits.shape[-1]
        self.num_detections, self.num_objects = assign_logits.shape

        # Clamp to avoid NANs.
        exists_logits = exists_logits.clamp(min=-40, max=40)
        assign_logits = assign_logits.clamp(min=-40, max=40)

        # This does all the work.
        if bp_iters is None:
            exists, assign = compute_marginals(exists_logits, assign_logits)
        else:
            exists, assign = compute_marginals_bp(
                exists_logits, assign_logits, bp_iters
            )

        # Wrap the results in Distribution objects.
        # This adds a final logit=0 element denoting spurious detection.
        padded_assign = torch.nn.functional.pad(assign, (0, 1), "constant", 0.0)
        self.assign_dist = dist.Categorical(logits=padded_assign)
        self.exists_dist = dist.Bernoulli(logits=exists)

