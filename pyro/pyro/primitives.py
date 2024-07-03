def param(
    name: str,
    init_tensor: Union[torch.Tensor, Callable[[], torch.Tensor], None] = None,
    constraint: constraints.Constraint = constraints.real,
    event_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Saves the variable as a parameter in the param store.
    To interact with the param store or write to disk,
    see `Parameters <parameters.html>`_.

    :param str name: name of parameter
    :param init_tensor: initial tensor or lazy callable that returns a tensor.
        For large tensors, it may be cheaper to write e.g.
        ``lambda: torch.randn(100000)``, which will only be evaluated on the
        initial statement.
    :type init_tensor: torch.Tensor or callable
    :param constraint: torch constraint, defaults to ``constraints.real``.
    :type constraint: torch.distributions.constraints.Constraint
    :param int event_dim: (optional) number of rightmost dimensions unrelated
        to batching. Dimension to the left of this will be considered batch
        dimensions; if the param statement is inside a subsampled plate, then
        corresponding batch dimensions of the parameter will be correspondingly
        subsampled. If unspecified, all dimensions will be considered event
        dims and no subsampling will be performed.
    :returns: A constrained parameter. The underlying unconstrained parameter
        is accessible via ``pyro.param(...).unconstrained()``, where
        ``.unconstrained`` is a weakref attribute.
    :rtype: torch.Tensor
    """
    # Note effectful(-) requires the double passing of name below.
    args = (name,) if init_tensor is None else (name, init_tensor)
    value = _param(*args, constraint=constraint, event_dim=event_dim, name=name)
    assert value is not None  # type narrowing guaranteed by _param
    return valuedef clear_param_store() -> None:
    """
    Clears the global :class:`~pyro.params.param_store.ParamStoreDict`.

    This is especially useful if you're working in a REPL. We recommend calling
    this before each training loop (to avoid leaking parameters from past
    models), and before each unit test (to avoid leaking parameters across
    tests).
    """
    _PYRO_PARAM_STORE.clear()class plate(PlateMessenger):
    """
    Construct for conditionally independent sequences of variables.

    ``plate`` can be used either sequentially as a generator or in parallel as
    a context manager (formerly ``irange`` and ``iarange``, respectively).

    Sequential :class:`plate` is similar to :py:func:`range` in that it generates
    a sequence of values.

    Vectorized :class:`plate` is similar to :func:`torch.arange` in that it
    yields an array of indices by which other tensors can be indexed.
    :class:`plate` differs from :func:`torch.arange` in that it also informs
    inference algorithms that the variables being indexed are conditionally
    independent.  To do this, :class:`plate` is a provided as context manager
    rather than a function, and users must guarantee that all computation
    within an :class:`plate` context is conditionally independent::

        with pyro.plate("name", size) as ind:
            # ...do conditionally independent stuff with ind...

    Additionally, :class:`plate` can take advantage of the conditional
    independence assumptions by subsampling the indices and informing inference
    algorithms to scale various computed values. This is typically used to
    subsample minibatches of data::

        with pyro.plate("data", len(data), subsample_size=100) as ind:
            batch = data[ind]
            assert len(batch) == 100

    By default ``subsample_size=False`` and this simply yields a
    ``torch.arange(0, size)``. If ``0 < subsample_size <= size`` this yields a
    single random batch of indices of size ``subsample_size`` and scales all
    log likelihood terms by ``size/batch_size``, within this context.

    .. warning::  This is only correct if all computation is conditionally
        independent within the context.

    :param str name: A unique name to help inference algorithms match
        :class:`plate` sites between models and guides.
    :param int size: Optional size of the collection being subsampled
        (like `stop` in builtin `range`).
    :param int subsample_size: Size of minibatches used in subsampling.
        Defaults to `size`.
    :param subsample: Optional custom subsample for user-defined subsampling
        schemes. If specified, then `subsample_size` will be set to
        `len(subsample)`.
    :type subsample: Anything supporting `len()`.
    :param int dim: An optional dimension to use for this independence index.
        If specified, ``dim`` should be negative, i.e. should index from the
        right. If not specified, ``dim`` is set to the rightmost dim that is
        left of all enclosing ``plate`` contexts.
    :param bool use_cuda: DEPRECATED, use the `device` arg instead.
        Optional bool specifying whether to use cuda tensors for `subsample`
        and `log_prob`. Defaults to ``torch.Tensor.is_cuda``.
    :param str device: Optional keyword specifying which device to place
        the results of `subsample` and `log_prob` on. By default, results
        are placed on the same device as the default tensor.
    :return: A reusabe context manager yielding a single 1-dimensional
        :class:`torch.Tensor` of indices.

    Examples:

        .. doctest::
           :hide:

           >>> loc, scale = torch.tensor(0.), torch.tensor(1.)
           >>> data = torch.randn(100)
           >>> z = dist.Bernoulli(0.5).sample((100,))

        >>> # This version declares sequential independence and subsamples data:
        >>> for i in pyro.plate('data', 100, subsample_size=10):
        ...     if z[i]:  # Control flow in this example prevents vectorization.
        ...         obs = pyro.sample(f'obs_{i}', dist.Normal(loc, scale),
        ...                           obs=data[i])

        >>> # This version declares vectorized independence:
        >>> with pyro.plate('data'):
        ...     obs = pyro.sample('obs', dist.Normal(loc, scale), obs=data)

        >>> # This version subsamples data in vectorized way:
        >>> with pyro.plate('data', 100, subsample_size=10) as ind:
        ...     obs = pyro.sample('obs', dist.Normal(loc, scale), obs=data[ind])

        >>> # This wraps a user-defined subsampling method for use in pyro:
        >>> ind = torch.randint(0, 100, (10,)).long() # custom subsample
        >>> with pyro.plate('data', 100, subsample=ind):
        ...     obs = pyro.sample('obs', dist.Normal(loc, scale), obs=data[ind])

        >>> # This reuses two different independence contexts.
        >>> x_axis = pyro.plate('outer', 320, dim=-1)
        >>> y_axis = pyro.plate('inner', 200, dim=-2)
        >>> with x_axis:
        ...     x_noise = pyro.sample("x_noise", dist.Normal(loc, scale))
        ...     assert x_noise.shape == (320,)
        >>> with y_axis:
        ...     y_noise = pyro.sample("y_noise", dist.Normal(loc, scale))
        ...     assert y_noise.shape == (200, 1)
        >>> with x_axis, y_axis:
        ...     xy_noise = pyro.sample("xy_noise", dist.Normal(loc, scale))
        ...     assert xy_noise.shape == (200, 320)

    See `SVI Part II <http://pyro.ai/examples/svi_part_ii.html>`_ for an
    extended discussion.
    """

    pass