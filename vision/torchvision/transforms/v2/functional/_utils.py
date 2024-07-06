def _get_kernel(functional, input_type, *, allow_passthrough=False):
    registry = _KERNEL_REGISTRY.get(functional)
    if not registry:
        raise ValueError(f"No kernel registered for functional {functional.__name__}.")

    for cls in input_type.__mro__:
        if cls in registry:
            return registry[cls]
        elif cls is tv_tensors.TVTensor:
            # We don't want user-defined tv_tensors to dispatch to the pure Tensor kernels, so we explicit stop the
            # MRO traversal before hitting torch.Tensor. We can even stop at tv_tensors.TVTensor, since we don't
            # allow kernels to be registered for tv_tensors.TVTensor anyway.
            break

    if allow_passthrough:
        return lambda inpt, *args, **kwargs: inpt

    raise TypeError(
        f"Functional F.{functional.__name__} supports inputs of type {registry.keys()}, "
        f"but got {input_type} instead."
    )

