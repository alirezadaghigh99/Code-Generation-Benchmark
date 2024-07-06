def sgd(
    lr: ScalarOrSchedule,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    *,
    moment_requires_grad: bool = False,
    maximize: bool = False,
) -> GradientTransformation:
    """Create a functional version of the canonical Stochastic Gradient Descent optimizer.

    This implements stochastic gradient descent. It also includes support for momentum, and nesterov
    acceleration, as these are standard practice when using stochastic gradient descent to train
    deep neural networks.

    References:
        - Sutskever et al., 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

    Args:
        lr (float or callable): This is a fixed global scaling factor or a learning rate
            scheduler.
        momentum (float, optional): The decay rate used by the momentum term. The momentum is not
            used when it is set to :const:`0.0`. (default: :const:`0.0`)
        weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
            (default: :const:`0.0`)
        dampening (float, optional): Dampening for momentum. (default: :const:`0.0`)
        nesterov (bool, optional): Whether to use Nesterov momentum. (default: :data:`False`)
        moment_requires_grad (bool, optional): If :data:`True` the momentums will be created with
            flag ``requires_grad=True``, this flag is often used in Meta-Learning algorithms.
            (default: :data:`False`)
        maximize (bool, optional): Maximize the params based on the objective, instead of
            minimizing. (default: :data:`False`)

    Returns:
        The corresponding :class:`GradientTransformation` instance.

    See Also:
        The functional optimizer wrapper :class:`torchopt.FuncOptimizer`.
    """
    # pylint: disable=unneeded-not
    if not (callable(lr) or lr >= 0.0):  # pragma: no cover
        raise ValueError(f'Invalid learning rate: {lr}')
    if not momentum >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid momentum value: {momentum}')
    if not weight_decay >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    if nesterov and (momentum <= 0.0 or dampening != 0.0):  # pragma: no cover
        raise ValueError('Nesterov momentum requires a momentum and zero dampening')
    # pylint: enable=unneeded-not

    chain_fn = chain
    flip_sign_and_add_weight_decay_fn = flip_sign_and_add_weight_decay
    trace_fn = trace
    scale_by_neg_lr_fn = scale_by_neg_lr

    if _get_use_chain_flat():  # default behavior
        chain_fn = chain_fn.flat  # type: ignore[attr-defined]
        flip_sign_and_add_weight_decay_fn = flip_sign_and_add_weight_decay_fn.flat  # type: ignore[attr-defined]
        trace_fn = trace_fn.flat  # type: ignore[attr-defined]
        scale_by_neg_lr_fn = scale_by_neg_lr_fn.flat  # type: ignore[attr-defined]

    return chain_fn(
        flip_sign_and_add_weight_decay_fn(weight_decay=weight_decay, maximize=maximize),
        trace_fn(
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            moment_requires_grad=moment_requires_grad,
        ),
        scale_by_neg_lr_fn(lr),
    )

