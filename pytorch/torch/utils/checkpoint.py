def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn't take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.4 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. In version 2.4 we will raise an exception "
            "if use_reentrant is not passed. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.",
            stacklevel=2
        )
        use_reentrant = True

    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret

def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.

    Use this with `torch.utils.checkpoint.checkpoint` to control which
    operations are recomputed during the backward pass.

    Args:
        policy_fn_or_list (Callable or List):
          - If a policy function is provided, it should accept a
            :class:`SelectiveCheckpointContext`, the :class:`OpOverload`, args and
            kwargs to the op, and return a :class:`CheckpointPolicy` enum value
            indicating whether the execution of the op should be recomputed or not.
          - If a list of operations is provided, it is equivalent to a policy
            returning `CheckpointPolicy.MUST_SAVE` for the specified
            operations and `CheckpointPolicy.PREFER_RECOMPUTE` for all other
            operations.
        allow_cache_entry_mutation (bool, optional): By default, an error is
            raised if any tensors cached by selective activation checkpoint are
            mutated in order to ensure correctness. If set to `True`, this check
            is disabled.
    Returns:
        A tuple of two context managers.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>> import functools
        >>>
        >>> x = torch.rand(10, 10, requires_grad=True)
        >>> y = torch.rand(10, 10, requires_grad=True)
        >>>
        >>> ops_to_save = [
        >>>    torch.ops.aten.mm.default,
        >>> ]
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    if op in ops_to_save:
        >>>        return CheckpointPolicy.MUST_SAVE
        >>>    else:
        >>>        return CheckpointPolicy.PREFER_RECOMPUTE
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> # or equivalently
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        >>>
        >>> def fn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    # NB: If grad_mode is disabled, checkpoint would not run forward under
    #     context_fn anyway, so proceed as usual.
    if isinstance(policy_fn_or_list, list):
        for op in policy_fn_or_list:
            if not isinstance(op, torch._ops.OpOverload):
                _extra_msg = (
                    "Please update the OpOverloadPacket to a specific OpOverload."
                    "For example, if you have `torch.ops.aten.mm`, change it to `torch.ops.aten.mm.default`."
                ) if isinstance(op, torch._ops.OpOverloadPacket) else ""
                raise ValueError(
                    f"Expected op in `op_list` to be an OpOverload but got: {op} "
                    f"of type {type(op)}. {_extra_msg}"
                )

        def policy_fn(ctx, op, *args, **kwargs):
            if op in policy_fn_or_list:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage: Dict[Any, List[Any]] = defaultdict(list)
    return (
        _CachingTorchDispatchMode(policy_fn, storage),
        _CachedTorchDispatchMode(policy_fn, storage, allow_cache_entry_mutation),
    )

def set_checkpoint_early_stop(enable: bool):
    """Context manager that sets whether checkpoint should stop recomputation early.

    By default, non-reentrant checkpoint stops recomputation as soon as it
    has computed all needed Tensors. This context manager can be used to disable
    that feature if it is problematic for your specific application.

    This context manager only needs to be active when forward is run. It does
    not need to be active during backward.

    Example::

    >>> # xdoctest: +SKIP(failing)
    >>> message = "saved tensors default hooks are disabled"
    >>> with set_checkpoint_early_stop(False):
    ...     # Any checkpoint under this context manager will respect this
    ...     # context manager, even if its backward is performed outside.
    ...     out = checkpoint(fn, inputs)
    ...
    >>> out.backward()
    """
    global _enable_checkpoint_early_stop
    try:
        prev = _enable_checkpoint_early_stop
        _enable_checkpoint_early_stop = enable
        yield
    finally:
        _enable_checkpoint_early_stop = prev

