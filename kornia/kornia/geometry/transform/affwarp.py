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

class Shear(nn.Module):
    r"""Shear the tensor.

    Args:
        shear: tensor containing the angle to shear
          in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The skewed tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> shear_factor = torch.tensor([[0.5, 0.0]])
        >>> out = Shear(shear_factor)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """

    def __init__(
        self, shear: torch.Tensor, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True
    ) -> None:
        super().__init__()
        self.shear: torch.Tensor = shear
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return shear(input, self.shear, self.mode, self.padding_mode, self.align_corners)

class Scale(nn.Module):
    r"""Scale the tensor by a factor.

    Args:
        scale_factor: The scale factor apply. The tensor
          must have a shape of (B) or (B, 2), where B is batch size.
          If (B), isotropic scaling will perform.
          If (B, 2), x-y-direction specific scaling will perform.
        center: The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The scaled tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> scale_factor = torch.tensor([[2., 2.]])
        >>> out = Scale(scale_factor)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """

    def __init__(
        self,
        scale_factor: torch.Tensor,
        center: Union[None, torch.Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.scale_factor: torch.Tensor = scale_factor
        self.center: Union[None, torch.Tensor] = center
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return scale(input, self.scale_factor, self.center, self.mode, self.padding_mode, self.align_corners)

class Affine(nn.Module):
    r"""Apply multiple elementary affine transforms simultaneously.

    Args:
        angle: Angle in degrees for counter-clockwise rotation around the center. The tensor
            must have a shape of (B), where B is the batch size.
        translation: Amount of pixels for translation in x- and y-direction. The tensor must
            have a shape of (B, 2), where B is the batch size and the last dimension contains dx and dy.
        scale_factor: Factor for scaling. The tensor must have a shape of (B), where B is the
            batch size.
        shear: Angles in degrees for shearing in x- and y-direction around the center. The
            tensor must have a shape of (B, 2), where B is the batch size and the last dimension contains sx and sy.
        center: Transformation center in pixels. The tensor must have a shape of (B, 2), where
            B is the batch size and the last dimension contains cx and cy. Defaults to the center of image to be
            transformed.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Raises:
        RuntimeError: If not one of ``angle``, ``translation``, ``scale_factor``, or ``shear`` is set.

    Returns:
        The transformed tensor with same shape as input.

    Example:
        >>> img = torch.rand(1, 2, 3, 5)
        >>> angle = 90. * torch.rand(1)
        >>> out = Affine(angle)(img)
        >>> print(out.shape)
        torch.Size([1, 2, 3, 5])
    """

    def __init__(
        self,
        angle: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        scale_factor: Optional[torch.Tensor] = None,
        shear: Optional[torch.Tensor] = None,
        center: Optional[torch.Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        batch_sizes = [arg.size()[0] for arg in (angle, translation, scale_factor, shear) if arg is not None]
        if not batch_sizes:
            msg = (
                "Affine was created without any affine parameter. At least one of angle, translation, scale_factor, or "
                "shear has to be set."
            )
            raise RuntimeError(msg)

        batch_size = batch_sizes[0]
        if not all(other == batch_size for other in batch_sizes[1:]):
            raise RuntimeError(f"The batch sizes of the affine parameters mismatch: {batch_sizes}")

        self._batch_size = batch_size

        super().__init__()
        device, dtype = _extract_device_dtype([angle, translation, scale_factor])

        if angle is None:
            angle = zeros(batch_size, device=device, dtype=dtype)
        self.angle = angle

        if translation is None:
            translation = zeros(batch_size, 2, device=device, dtype=dtype)
        self.translation = translation

        if scale_factor is None:
            scale_factor = ones(batch_size, 2, device=device, dtype=dtype)
        self.scale_factor = scale_factor

        self.shear = shear
        self.center = center
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.shear is None:
            sx = sy = None
        else:
            sx, sy = self.shear[..., 0], self.shear[..., 1]

        if self.center is None:
            center = _compute_tensor_center(input).expand(input.size()[0], -1)
        else:
            center = self.center

        matrix = get_affine_matrix2d(self.translation, center, self.scale_factor, -self.angle, sx=sx, sy=sy)
        return affine(input, matrix[..., :2, :3], self.mode, self.padding_mode, self.align_corners)

class Rotate(nn.Module):
    r"""Rotate the tensor anti-clockwise about the centre.

    Args:
        angle: The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The rotated tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> angle = torch.tensor([90.])
        >>> out = Rotate(angle)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """

    def __init__(
        self,
        angle: torch.Tensor,
        center: Union[None, torch.Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.angle: torch.Tensor = angle
        self.center: Union[None, torch.Tensor] = center
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rotate(input, self.angle, self.center, self.mode, self.padding_mode, self.align_corners)

class Translate(nn.Module):
    r"""Translate the tensor in pixel units.

    Args:
        translation: tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The translated tensor with the same shape as the input.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> translation = torch.tensor([[1., 0.]])
        >>> out = Translate(translation)(img)
        >>> print(out.shape)
        torch.Size([1, 3, 4, 4])
    """

    def __init__(
        self, translation: torch.Tensor, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True
    ) -> None:
        super().__init__()
        self.translation: torch.Tensor = translation
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return translate(input, self.translation, self.mode, self.padding_mode, self.align_corners)

class Resize(nn.Module):
    r"""Resize the input torch.Tensor to the given size.

    Args:
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
        The resized tensor with the shape of the given size.

    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = Resize((6, 8))(img)
        >>> print(out.shape)
        torch.Size([1, 3, 6, 8])

    .. raw:: html

        <gradio-app space="kornia/kornia-resize-antialias"></gradio-app>
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
        align_corners: Optional[bool] = None,
        side: str = "short",
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.size: Union[int, Tuple[int, int]] = size
        self.interpolation: str = interpolation
        self.align_corners: Optional[bool] = align_corners
        self.side: str = side
        self.antialias: bool = antialias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return resize(
            input,
            self.size,
            self.interpolation,
            align_corners=self.align_corners,
            side=self.side,
            antialias=self.antialias,
        )

