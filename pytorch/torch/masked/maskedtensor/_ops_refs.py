def ones_like(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1)
    result_data = func(_get_data(args[0]), **kwargs)
    return MaskedTensor(result_data, _maybe_get_mask(args[0]))

def _softmax_backward_data(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=4)
    grad, output, dim, input_dtype = args
    if is_masked_tensor(grad) and is_masked_tensor(output):
        if not _masks_match(grad, output):
            raise ValueError(
                "__torch_dispatch__, {func}: expected the masks of grad and output to match"
            )
        grad_data = _get_data(grad)
        new_grad_data = torch.ops.aten._masked_softmax_backward(
            grad_data,
            _get_data(output),
            ~_maybe_get_mask(grad),
            dim % grad_data.ndim,
        )
        res = MaskedTensor(new_grad_data, _maybe_get_mask(grad))
        return res
    else:
        raise ValueError(
            f"__torch_dispatch__, {func}: grad and output must both be MaskedTensors"
        )

