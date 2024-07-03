def ones_like(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1)
    result_data = func(_get_data(args[0]), **kwargs)
    return MaskedTensor(result_data, _maybe_get_mask(args[0]))