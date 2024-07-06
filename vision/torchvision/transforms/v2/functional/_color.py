def grayscale_to_rgb(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.GrayscaleToRgb` for details."""
    if torch.jit.is_scripting():
        return grayscale_to_rgb_image(inpt)

    _log_api_usage_once(grayscale_to_rgb)

    kernel = _get_kernel(grayscale_to_rgb, type(inpt))
    return kernel(inpt)

