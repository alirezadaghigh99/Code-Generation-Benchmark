def rescale(
    input: torch.Tensor,
    factor: Union[float, Tuple[float, float]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    antialias: bool = False,
) -> torch.Tensor:
    r"""Rescale the input torch.Tensor with the given factor.

    .. image:: _static/img/rescale.png

    Args:
        input: The image tensor to be scale with shape of :math:`(B, C, H, W)`.
        factor: Desired scaling factor in each direction. If scalar, the value is used
            for both the x- and y-direction.
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            ``'bicubic'`` | ``'trilinear'`` | ``'area'``.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The rescaled tensor with the shape as the specified size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = rescale(img, (2, 3))
        >>> print(out.shape)
        torch.Size([1, 3, 8, 12])
    """
    if isinstance(factor, float):
        factor_vert = factor_horz = factor
    else:
        factor_vert, factor_horz = factor

    height, width = input.size()[-2:]
    size = (int(height * factor_vert), int(width * factor_horz))
    return resize(input, size, interpolation=interpolation, align_corners=align_corners, antialias=antialias)

def resize(
    input: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bilinear",
    align_corners: Optional[bool] = None,
    side: str = "short",
    antialias: bool = False,
) -> torch.Tensor:
    r"""Resize the input torch.Tensor to the given size.

    .. image:: _static/img/resize.png

    Args:
        tensor: The image tensor to be skewed with shape of :math:`(..., H, W)`.
            `...` means there can be any number of dimensions.
        size: Desired output size. If size is a sequence like (h, w),
            output size will be matched to this. If size is an int, smaller edge of the image will
            be matched to this number. i.e, if height > width, then image will be rescaled
            to (size * height / width, size)
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            'bicubic' | 'trilinear' | 'area'.
        align_corners: interpolation flag.
        side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
            or ``'horz'``.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.

    Returns:
        The resized tensor with the shape as the specified size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = resize(img, (6, 8))
        >>> print(out.shape)
        torch.Size([1, 3, 6, 8])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) < 2:
        raise ValueError(f"Input tensor must have at least two dimensions. Got {len(input.shape)}")

    input_size = h, w = input.shape[-2:]
    if isinstance(size, int):
        aspect_ratio = w / h
        size = _side_to_image_size(size, aspect_ratio, side)

    if size == input_size:
        return input

    factors = (h / size[0], w / size[1])

    # We do bluring only for downscaling
    antialias = antialias and (max(factors) > 1)

    if antialias:
        # First, we have to determine sigma
        # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
        sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))

        # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
        # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

        # Make sure it is odd
        if (ks[0] % 2) == 0:
            ks = ks[0] + 1, ks[1]

        if (ks[1] % 2) == 0:
            ks = ks[0], ks[1] + 1

        input = gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output

