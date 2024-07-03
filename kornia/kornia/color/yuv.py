def rgb_to_yuv420(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Convert an RGB image to YUV 420 (subsampled).

    The image data is assumed to be in the range of (0, 1). Input need to be padded to be evenly divisible by 2
    horizontal and vertical. This function will output chroma siting (0.5,0.5)

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        A Tensor containing the Y plane with shape :math:`(*, 1, H, W)`
        A Tensor containing the UV planes with shape :math:`(*, 2, H/2, W/2)`

    Example:
        >>> input = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # (2x1x4x6, 2x2x2x3)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    yuvimage = rgb_to_yuv(image)

    return (yuvimage[..., :1, :, :], yuvimage[..., 1:3, :, :].unfold(-2, 2, 2).unfold(-2, 2, 2).mean((-1, -2)))def yuv420_to_rgb(imagey: torch.Tensor, imageuv: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV420 image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Input need to be padded to be evenly divisible by 2 horizontal and vertical.
    This function assumed chroma siting is (0.5, 0.5)

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`(*, 1, H, W)`.
        imageuv: UV (chroma) Image planes to be converted to RGB with shape :math:`(*, 2, H/2, W/2)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> inputy = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> output = yuv420_to_rgb(inputy, inputuv)  # 2x3x4x6
    """
    if not isinstance(imagey, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(imagey)}")

    if not isinstance(imageuv, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(imageuv)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imageuv.shape) < 3 or imageuv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of (*, 2, H/2, W/2). Got {imageuv.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if (
        len(imageuv.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] / imageuv.shape[-2] != 2
        or imagey.shape[-1] / imageuv.shape[-1] != 2
    ):
        raise ValueError(
            f"Input imageuv H&W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat([imagey, imageuv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)], dim=-3)
    # then convert the yuv444 tensor

    return yuv_to_rgb(yuv444image)