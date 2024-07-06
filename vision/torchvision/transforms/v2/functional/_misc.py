def gaussian_noise(inpt: torch.Tensor, mean: float = 0.0, sigma: float = 0.1, clip: bool = True) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.GaussianNoise`"""
    if torch.jit.is_scripting():
        return gaussian_noise_image(inpt, mean=mean, sigma=sigma)

    _log_api_usage_once(gaussian_noise)

    kernel = _get_kernel(gaussian_noise, type(inpt))
    return kernel(inpt, mean=mean, sigma=sigma, clip=clip)

