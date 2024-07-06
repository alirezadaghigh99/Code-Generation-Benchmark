def rgb_to_hls(image: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Convert an RGB image to HLS.

    .. image:: _static/img/rgb_to_hls.png

    The image data is assumed to be in the range of (0, 1).

    NOTE: this method cannot be compiled with JIT in pytohrch < 1.7.0

    Args:
        image: RGB image to be converted to HLS with shape :math:`(*, 3, H, W)`.
        eps: epsilon value to avoid div by zero.

    Returns:
        HLS version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hls(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    _RGB2HSL_IDX = tensor([[[0.0]], [[1.0]], [[2.0]]], device=image.device, dtype=image.dtype)  # 3x1x1

    _img_max: Tuple[Tensor, Tensor] = image.max(-3)
    maxc = _img_max[0]
    imax = _img_max[1]
    minc: Tensor = image.min(-3)[0]

    if image.requires_grad:
        l_ = maxc + minc
        s = maxc - minc
        # weird behaviour with undefined vars in JIT...
        # scripting requires image_hls be defined even if it is not used :S
        h = l_  # assign to any tensor...
        image_hls = l_  # assign to any tensor...
    else:
        # define the resulting image to avoid the torch.stack([h, l, s])
        # so, h, l and s require inplace operations
        # NOTE: stack() increases in a 10% the cost in colab
        image_hls = torch.empty_like(image)
        h, l_, s = image_hls[..., 0, :, :], image_hls[..., 1, :, :], image_hls[..., 2, :, :]
        torch.add(maxc, minc, out=l_)  # l = max + min
        torch.sub(maxc, minc, out=s)  # s = max - min

    # precompute image / (max - min)
    im = image / (s + eps).unsqueeze(-3)

    # epsilon cannot be inside the torch.where to avoid precision issues
    s /= where(l_ < 1.0, l_, 2.0 - l_) + eps  # saturation
    l_ /= 2  # luminance

    # note that r,g and b were previously div by (max - min)
    r, g, b = im[..., 0, :, :], im[..., 1, :, :], im[..., 2, :, :]
    # h[imax == 0] = (((g - b) / (max - min)) % 6)[imax == 0]
    # h[imax == 1] = (((b - r) / (max - min)) + 2)[imax == 1]
    # h[imax == 2] = (((r - g) / (max - min)) + 4)[imax == 2]
    cond = imax[..., None, :, :] == _RGB2HSL_IDX
    if image.requires_grad:
        h = ((g - b) % 6) * cond[..., 0, :, :]
    else:
        # replacing `torch.mul` with `out=h` with python * operator gives wrong results
        torch.mul((g - b) % 6, cond[..., 0, :, :], out=h)
    h += (b - r + 2) * cond[..., 1, :, :]
    h += (r - g + 4) * cond[..., 2, :, :]
    # h = 2.0 * math.pi * (60.0 * h) / 360.0
    h *= math.pi / 3.0  # hue [0, 2*pi]

    if image.requires_grad:
        return stack([h, l_, s], -3)
    return image_hls

