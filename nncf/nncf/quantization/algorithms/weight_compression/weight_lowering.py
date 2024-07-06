def reshape_weight_for_grouped_quantization(
    weight: Tensor, reduction_axes: ReductionAxes, group_size: int
) -> Tuple[Tensor, int]:
    """
    Reshapes weight for group-wise quantization and return a new reduction axis for collecting statistics per group
    dimension. Having weight with shapes [c_out, c_in] and group size = 128, shape of reshaped weight is
    [c_out, c_in // 128, 128].

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weight and new reduction axis.
    """
    assert group_size != -1
    if isinstance(reduction_axes, tuple) and len(reduction_axes) == 1:
        reduction_axes = reduction_axes[0]
    if not isinstance(reduction_axes, int):
        raise NotImplementedError(
            f"Group-wise quantization expects a single reduction axis, but given: {reduction_axes}."
        )
    channel_size = weight.shape[reduction_axes]
    if channel_size % group_size != 0:
        raise nncf.ValidationError(f"Channel size {channel_size} should be divisible by size of group {group_size}")

    num_groups_per_channel = channel_size // group_size
    shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axes : reduction_axes + 1] = (num_groups_per_channel, group_size)
    reshaped_weight = weight.reshape(shape)
    reduction_axes += 1
    return reshaped_weight, reduction_axes

