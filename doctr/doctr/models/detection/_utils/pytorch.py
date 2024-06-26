def dilate(x: Tensor, kernel_size: int) -> Tensor:
    """Performs dilation on a given tensor

    Args:
    ----
        x: boolean tensor of shape (N, C, H, W)
        kernel_size: the size of the kernel to use for dilation

    Returns:
    -------
        the dilated tensor
    """
    _pad = (kernel_size - 1) // 2

    return max_pool2d(x, kernel_size, stride=1, padding=_pad)