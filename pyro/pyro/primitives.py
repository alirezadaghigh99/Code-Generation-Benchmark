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
    return value

def clear_param_store() -> None:
    """
    Clears the global :class:`~pyro.params.param_store.ParamStoreDict`.

    This is especially useful if you're working in a REPL. We recommend calling
    this before each training loop (to avoid leaking parameters from past
    models), and before each unit test (to avoid leaking parameters across
    tests).
    """
    _PYRO_PARAM_STORE.clear()

