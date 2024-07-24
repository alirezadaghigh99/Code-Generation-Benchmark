def rgb_to_rgba(image: torch.Tensor, alpha_val: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Convert an image from RGB to RGBA.

    Args:
        image: RGB Image to be converted to RGBA of shape :math:`(*,3,H,W)`.
        alpha_val (float, torch.Tensor): A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        RGBA version of the image with shape :math:`(*,4,H,W)`.

    .. note:: The current functionality is NOT supported by Torchscript.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_rgba(input, 1.) # 2x4x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)

    a: torch.Tensor = cast(torch.Tensor, alpha_val)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))

    return torch.cat([r, g, b, a], dim=-3)

class BgrToRgba(nn.Module):
    r"""Convert an image from BGR to RGBA.

    Add an alpha channel to existing RGB image.

    Args:
        alpha_val: A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        RGBA version of the image with shape :math:`(*,4,H,W)`.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 4, H, W)`

    .. note:: The current functionality is NOT supported by Torchscript.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgba = BgrToRgba(1.)
        >>> output = rgba(input)  # 2x4x4x5
    """

    def __init__(self, alpha_val: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.alpha_val = alpha_val

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_rgba(image, self.alpha_val)

class RgbaToRgb(nn.Module):
    r"""Convert an image from RGBA to RGB.

    Remove an alpha channel from RGB image.

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToRgb()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgba_to_rgb(image)

class RgbaToBgr(nn.Module):
    r"""Convert an image from RGBA to BGR.

    Remove an alpha channel from BGR image.

    Returns:
        BGR version of the image.

    Shape:
        - image: :math:`(*, 4, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 4, 5)
        >>> rgba = RgbaToBgr()
        >>> output = rgba(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgba_to_bgr(image)

class LinearRgbToRgb(nn.Module):
    r"""Convert a linear RGB image to sRGB.

    Applies gamma correction to linear RGB values, at the end of colorspace conversions, to get sRGB.

    Returns:
        sRGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> srgb = LinearRgbToRgb()
        >>> output = srgb(input)  # 2x3x4x5

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb

        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

        [3] https://en.wikipedia.org/wiki/SRGB
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return linear_rgb_to_rgb(image)

