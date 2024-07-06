def custom_op(qualname, func_or_schema=None):
    r"""Register a new custom operator

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
      various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs.

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Arguments:
        qualname (str): Should be a string that looks like
            "namespace::operator_name". Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        func_or_schema (Union[Callable, str]): Each PyTorch operator needs a
            schema that tells PyTorch the types of the inputs/outputs.
            If this is a Callable, we will automatically infer the schema from
            the type annotations on the function (see examples). Otherwise,
            if you don't want to use type annotations, you may provide us the
            schema string.

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Step 1: define the custom op.
        >>> # We need to provide the API a "prototype function"
        >>> # (a function that returns NotImplementedError), from which
        >>> # we will infer the types of the inputs and outputs.
        >>> @torch._custom_ops.custom_op("mylibrary::numpy_sin")
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     raise NotImplementedError
        >>>
        >>> # The custom op is now accessible via the torch.ops module:
        >>> torch.ops.mylibrary.numpy_sin
        >>>
        >>> # Step 2: Register an implementation for various PyTorch subsystems
        >>>
        >>> # Register an implementation for CPU tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cpu")
        >>> def numpy_sin_impl_cpu(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Register an implementation for CUDA tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cuda")
        >>> def numpy_sin_impl_cuda(x):
        >>>     return torch.from_numpy(np.sin(x.cpu().numpy())).to(x.device)
        >>>
        >>> x = torch.randn(3)
        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cpu
        >>>
        >>> x_cuda = x.cuda()
        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cuda

    """
    ns, name = parse_qualname(qualname)
    validate_namespace(ns)

    def inner(func):
        if not inspect.isfunction(func):
            raise ValueError(
                f"custom_op(...)(func): Expected `func` to be a Python "
                f"function, got: {type(func)}"
            )

        if func.__name__ != name:
            raise ValueError(
                f"custom_op(qualname='{qualname}', ...)(func): expected `func` "
                f"to have name '{name}' but got '{func.__name__}'. "
                f"Please either change the name of `func` or the qualname that "
                f"is passed to `custom_op`"
            )

        schema = infer_schema(func)
        _custom_op_with_schema(qualname, schema)
        return func

    if func_or_schema is None:
        return inner
    if isinstance(func_or_schema, str):
        _custom_op_with_schema(qualname, func_or_schema)
    else:
        return inner(func_or_schema)

