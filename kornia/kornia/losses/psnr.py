class PSNRLoss(nn.Module):
    r"""Create a criterion that calculates the PSNR loss.

    The loss is computed as follows:

     .. math::

        \text{loss} = -\text{psnr(x, y)}

    See :meth:`~kornia.losses.psnr` for details abut PSNR.

    Args:
        max_val: The maximum value in the image tensor.

    Shape:
        - Image: arbitrary dimensional tensor :math:`(*)`.
        - Target: arbitrary dimensional tensor :math:`(*)` same shape as image.
        - Output: a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> criterion = PSNRLoss(2.)
        >>> criterion(ones, 1.2 * ones) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(-20.0000)
    """

    def __init__(self, max_val: float) -> None:
        super().__init__()
        self.max_val: float = max_val

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr_loss(image, target, self.max_val)

