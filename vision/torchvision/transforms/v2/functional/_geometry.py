def resize_image(
    image: torch.Tensor,
    size: Optional[List[int]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = True,
) -> torch.Tensor:
    interpolation = _check_interpolation(interpolation)
    antialias = False if antialias is None else antialias
    align_corners: Optional[bool] = None
    if interpolation == InterpolationMode.BILINEAR or interpolation == InterpolationMode.BICUBIC:
        align_corners = False
    else:
        # The default of antialias is True from 0.17, so we don't warn or
        # error if other interpolation modes are used. This is documented.
        antialias = False

    shape = image.shape
    numel = image.numel()
    num_channels, old_height, old_width = shape[-3:]
    new_height, new_width = _compute_resized_output_size((old_height, old_width), size=size, max_size=max_size)

    if (new_height, new_width) == (old_height, old_width):
        return image
    elif numel > 0:
        dtype = image.dtype
        acceptable_dtypes = [torch.float32, torch.float64]
        if interpolation == InterpolationMode.NEAREST or interpolation == InterpolationMode.NEAREST_EXACT:
            # uint8 dtype can be included for cpu and cuda input if nearest mode
            acceptable_dtypes.append(torch.uint8)
        elif image.device.type == "cpu":
            if _do_native_uint8_resize_on_cpu(interpolation):
                acceptable_dtypes.append(torch.uint8)

        image = image.reshape(-1, num_channels, old_height, old_width)
        strides = image.stride()
        if image.is_contiguous(memory_format=torch.channels_last) and image.shape[0] == 1 and numel != strides[0]:
            # There is a weird behaviour in torch core where the output tensor of `interpolate()` can be allocated as
            # contiguous even though the input is un-ambiguously channels_last (https://github.com/pytorch/pytorch/issues/68430).
            # In particular this happens for the typical torchvision use-case of single CHW images where we fake the batch dim
            # to become 1CHW. Below, we restride those tensors to trick torch core into properly allocating the output as
            # channels_last, thus preserving the memory format of the input. This is not just for format consistency:
            # for uint8 bilinear images, this also avoids an extra copy (re-packing) of the output and saves time.
            # TODO: when https://github.com/pytorch/pytorch/issues/68430 is fixed (possibly by https://github.com/pytorch/pytorch/pull/100373),
            # we should be able to remove this hack.
            new_strides = list(strides)
            new_strides[0] = numel
            image = image.as_strided((1, num_channels, old_height, old_width), new_strides)

        need_cast = dtype not in acceptable_dtypes
        if need_cast:
            image = image.to(dtype=torch.float32)

        image = interpolate(
            image,
            size=[new_height, new_width],
            mode=interpolation.value,
            align_corners=align_corners,
            antialias=antialias,
        )

        if need_cast:
            if interpolation == InterpolationMode.BICUBIC and dtype == torch.uint8:
                # This path is hit on non-AVX archs, or on GPU.
                image = image.clamp_(min=0, max=255)
            if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                image = image.round_()
            image = image.to(dtype=dtype)

    return image.reshape(shape[:-3] + (num_channels, new_height, new_width))

