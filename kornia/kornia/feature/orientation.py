class PassLAF(nn.Module):
    """Dummy module to use instead of local feature orientation or affine shape estimator."""

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            laf: :math:`(B, N, 2, 3)`
            img: :math:`(B, 1, H, W)`

        Returns:
            LAF, unchanged :math:`(B, N, 2, 3)`
        """
        return laf

