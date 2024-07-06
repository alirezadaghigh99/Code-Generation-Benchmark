def get_weight_channel_axes(metatype: om.PTOperatorMetatype, ndims: int, input_port_id: int) -> Tuple[int, ...]:
    """
    Returns axes numbers of the weight tensor which correspond to its channels.

    :param metatype: The node metatype for which the target dimension is being determined.
    :param input_port_id: The input port id.
    :return: The target dimension for weight compression.
    """
    if metatype == om.PTAddmmMetatype:
        if input_port_id == 1:
            return (ndims - 2,)
        if input_port_id == 2:
            return (ndims - 1,)
        raise ValueError(f"Unexpected {input_port_id=} for {metatype=}")
    if metatype == om.PTMatMulMetatype:
        if input_port_id == 0:
            return () if ndims < 2 else (ndims - 2,)
        if input_port_id == 1:
            return () if ndims < 2 else (ndims - 1,)
        raise ValueError(f"Unexpected {input_port_id=} for {metatype=}")
    if metatype in [om.PTConvTranspose1dMetatype, om.PTConvTranspose2dMetatype, om.PTConvTranspose3dMetatype]:
        return (1,)
    return (0,)

