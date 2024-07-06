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

