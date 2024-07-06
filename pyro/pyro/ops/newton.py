def newton_step(loss, x, trust_radius=None):
    """
    Performs a Newton update step to minimize loss on a batch of variables,
    optionally constraining to a trust region [1].

    This is especially usful because the final solution of newton iteration
    is differentiable wrt the inputs, even when all but the final ``x`` is
    detached, due to this method's quadratic convergence [2]. ``loss`` must be
    twice-differentiable as a function of ``x``. If ``loss`` is ``2+d``-times
    differentiable, then the return value of this function is ``d``-times
    differentiable.

    When ``loss`` is interpreted as a negative log probability density, then
    the return values ``mode,cov`` of this function can be used to construct a
    Laplace approximation ``MultivariateNormal(mode,cov)``.

    .. warning:: Take care to detach the result of this function when used in
        an optimization loop. If you forget to detach the result of this
        function during optimization, then backprop will propagate through
        the entire iteration process, and worse will compute two extra
        derivatives for each step.

    Example use inside a loop::

        x = torch.zeros(1000, 2)  # arbitrary initial value
        for step in range(100):
            x = x.detach()          # block gradients through previous steps
            x.requires_grad = True  # ensure loss is differentiable wrt x
            loss = my_loss_function(x)
            x = newton_step(loss, x, trust_radius=1.0)
        # the final x is still differentiable

    [1] Yuan, Ya-xiang. Iciam. Vol. 99. 2000.
        "A review of trust region algorithms for optimization."
        ftp://ftp.cc.ac.cn/pub/yyx/papers/p995.pdf
    [2] Christianson, Bruce. Optimization Methods and Software 3.4 (1994)
        "Reverse accumulation and attractive fixed points."
        http://uhra.herts.ac.uk/bitstream/handle/2299/4338/903839.pdf

    :param torch.Tensor loss: A scalar function of ``x`` to be minimized.
    :param torch.Tensor x: A dependent variable of shape ``(N, D)``
        where ``N`` is the batch size and ``D`` is a small number.
    :param float trust_radius: An optional trust region trust_radius. The
        updated value ``mode`` of this function will be within
        ``trust_radius`` of the input ``x``.
    :return: A pair ``(mode, cov)`` where ``mode`` is an updated tensor
        of the same shape as the original value ``x``, and ``cov`` is an
        esitmate of the covariance DxD matrix with
        ``cov.shape == x.shape[:-1] + (D,D)``.
    :rtype: tuple
    """
    if x.dim() < 1:
        raise ValueError(
            "Expected x to have at least one dimension, actual shape {}".format(x.shape)
        )
    dim = x.shape[-1]
    if dim == 1:
        return newton_step_1d(loss, x, trust_radius)
    elif dim == 2:
        return newton_step_2d(loss, x, trust_radius)
    elif dim == 3:
        return newton_step_3d(loss, x, trust_radius)
    else:
        raise NotImplementedError("newton_step_nd is not implemented")

