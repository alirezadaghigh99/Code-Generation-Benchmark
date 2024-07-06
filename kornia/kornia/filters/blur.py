def box_blur(
    input: Tensor, kernel_size: tuple[int, int] | int, border_type: str = "reflect", separable: bool = False
) -> Tensor:
    r"""Blur an image using the box filter.

    .. image:: _static/img/box_blur.png

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        image: the image to blur with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = box_blur(input, (3, 3))  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    KORNIA_CHECK_IS_TENSOR(input)

    if separable:
        ky, kx = _unpack_2d_ks(kernel_size)
        kernel_y = get_box_kernel1d(ky, device=input.device, dtype=input.dtype)
        kernel_x = get_box_kernel1d(kx, device=input.device, dtype=input.dtype)
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel = get_box_kernel2d(kernel_size, device=input.device, dtype=input.dtype)
        out = filter2d(input, kernel, border_type)

    return out

