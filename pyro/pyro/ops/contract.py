def ubersum(equation, *operands, **kwargs):
    """
    Deprecated, use :func:`einsum` instead.
    """
    warnings.warn(
        "'ubersum' is deprecated, use 'pyro.ops.contract.einsum' instead",
        DeprecationWarning,
    )
    if "batch_dims" in kwargs:
        warnings.warn(
            "'batch_dims' is deprecated, use 'plates' instead", DeprecationWarning
        )
        kwargs["plates"] = kwargs.pop("batch_dims")
    kwargs.setdefault("backend", "pyro.ops.einsum.torch_log")
    return einsum(equation, *operands, **kwargs)