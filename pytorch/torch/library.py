def _scoped_library(*args, **kwargs):
    try:
        lib = Library(*args, **kwargs)
        yield lib
    finally:
        lib._destroy()def impl(qualname, types, func=None, *, lib=None):
    """Register an implementation for a device type for this operator.

    You may pass "default" for ``types`` to register this implementation as the
    default implementation for ALL device types.
    Please only use this if the implementation truly supports all device types;
    for example, this is true if it is a composition of built-in PyTorch operators.

    Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".

    Args:
        qualname (str): Should be a string that looks like "namespace::operator_name".
        types (str | Sequence[str]): The device types to register an impl to.
        lib (Optional[Library]): If provided, the lifetime of this registration
            will be tied to the lifetime of the Library object.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::mysin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the cpu device
        >>> @torch.library.impl("mylib::mysin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.mysin(x)
        >>> assert torch.allclose(y, x.sin())
    """
    if isinstance(types, str):
        types = (types,)
    keys = set({})
    for typ in types:
        is_dispatch_key = torch._C._parse_dispatch_key(typ)
        if is_dispatch_key:
            # We also support passing a DispatchKey to impl. Please prefer using
            # the higher-level torch.library APIs and only pass DispatchKey to
            # torch.library.impl with caution (or even better, don't use this
            # option and file an issue on GitHub for what you need).
            # We don't advertise this to users because
            # it is very easy to shoot yourself in the foot.
            keys.add(typ)
        else:
            keys.add(_device_type_to_key(typ))

    def register(func):
        namespace, _ = torch._library.utils.parse_namespace(qualname)
        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(use_lib)
        else:
            use_lib = lib
        for key in keys:
            use_lib.impl(qualname, func, key)

    if func is None:
        return register
    else:
        register(func)def impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1):
    r"""This API was renamed to :func:`torch.library.register_fake` in PyTorch 2.4.
    Please use that instead.
    """
    if func is not None:
        _stacklevel = _stacklevel + 1
    return register_fake(qualname, func, lib=lib, _stacklevel=_stacklevel)