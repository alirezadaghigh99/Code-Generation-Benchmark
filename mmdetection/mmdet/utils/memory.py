def cast_tensor_type(inputs, src_type=None, dst_type=None):
    """Recursively convert Tensor in inputs from ``src_type`` to ``dst_type``.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype | torch.device): Source type.
        src_type (torch.dtype | torch.device): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    assert dst_type is not None
    if isinstance(inputs, torch.Tensor):
        if isinstance(dst_type, torch.device):
            # convert Tensor to dst_device
            if hasattr(inputs, 'to') and \
                    hasattr(inputs, 'device') and \
                    (inputs.device == src_type or src_type is None):
                return inputs.to(dst_type)
            else:
                return inputs
        else:
            # convert Tensor to dst_dtype
            if hasattr(inputs, 'to') and \
                    hasattr(inputs, 'dtype') and \
                    (inputs.dtype == src_type or src_type is None):
                return inputs.to(dst_type)
            else:
                return inputs
        # we need to ensure that the type of inputs to be casted are the same
        # as the argument `src_type`.
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type=src_type, dst_type=dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type=src_type, dst_type=dst_type)
            for item in inputs)
    # TODO: Currently not supported
    # elif isinstance(inputs, InstanceData):
    #     for key, value in inputs.items():
    #         inputs[key] = cast_tensor_type(
    #             value, src_type=src_type, dst_type=dst_type)
    #     return inputs
    else:
        return inputs