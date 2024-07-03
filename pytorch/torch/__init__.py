    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtypedef compile(
    model: _Callable[_InputT, _RetT],
    *,
    fullgraph: builtins.bool = False,
    dynamic: _Optional[builtins.bool] = None,
    backend: _Union[str, _Callable] = "inductor",
    mode: _Union[str, None] = None,
    options: _Optional[_Dict[str, _Union[str, builtins.int, builtins.bool]]] = None,
    disable: builtins.bool = False,
) -> _Callable[_InputT, _RetT]:
    ...def is_tensor(obj: _Any, /) -> _TypeGuard["torch.Tensor"]:
    r"""Returns True if `obj` is a PyTorch tensor.

    Note that this function is simply doing ``isinstance(obj, Tensor)``.
    Using that ``isinstance`` check is better for typechecking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_tensor``.

    Args:
        obj (object): Object to test
    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> torch.is_tensor(x)
        True

    """
    return isinstance(obj, torch.Tensor)def typename(obj: _Any, /) -> str:
    """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        obj (object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> torch.typename(x)
        'torch.LongTensor'
        >>> torch.typename(torch.nn.Parameter)
        'torch.nn.parameter.Parameter'
    """
    if isinstance(obj, torch.Tensor):
        return obj.type()

    module = getattr(obj, "__module__", "") or ""
    qualname = ""

    if hasattr(obj, "__qualname__"):
        qualname = obj.__qualname__
    elif hasattr(obj, "__name__"):
        qualname = obj.__name__
    else:
        module = obj.__class__.__module__ or ""
        qualname = obj.__class__.__qualname__

    if module in {"", "builtins"}:
        return qualname
    return f"{module}.{qualname}"    def dtype(self):
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtypedef compile(
    model: _Callable[_InputT, _RetT],
    *,
    fullgraph: builtins.bool = False,
    dynamic: _Optional[builtins.bool] = None,
    backend: _Union[str, _Callable] = "inductor",
    mode: _Union[str, None] = None,
    options: _Optional[_Dict[str, _Union[str, builtins.int, builtins.bool]]] = None,
    disable: builtins.bool = False,
) -> _Callable[_InputT, _RetT]:
    ...