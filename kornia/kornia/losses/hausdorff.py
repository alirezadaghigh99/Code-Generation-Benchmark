class HausdorffERLoss(_HausdorffERLossBase):
    r"""Binary Hausdorff loss based on morphological erosion.

    Hausdorff Distance loss measures the maximum distance of a predicted segmentation boundary to
    the nearest ground-truth edge pixel. For two segmentation point sets X and Y ,
    the one-sided HD from X to Y is defined as:

    .. math::

        hd(X,Y) = \max_{x \in X} \min_{y \in Y}||x - y||_2

    Furthermore, the bidirectional HD is:

    .. math::

        HD(X,Y) = max(hd(X, Y), hd(Y, X))

    This is an Hausdorff Distance (HD) Loss that based on morphological erosion, which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
    blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.

    Examples:
        >>> hdloss = HausdorffERLoss()
        >>> input = torch.randn(5, 3, 20, 20)
        >>> target = (torch.rand(5, 1, 20, 20) * 2).long()
        >>> res = hdloss(input, target)
    """

    conv = torch.conv2d
    max_pool = nn.AdaptiveMaxPool2d(1)

    def get_kernel(self) -> Tensor:
        """Get kernel for image morphology convolution."""
        cross = tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        kernel = cross * 0.2
        return kernel[None]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Hausdorff loss.

        Args:
            pred: predicted tensor with a shape of :math:`(B, C, H, W)`.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target tensor with a shape of :math:`(B, 1, H, W)`.

        Returns:
            Estimated Hausdorff Loss.
        """
        if pred.dim() != 4:
            raise ValueError(f"Only 2D images supported. Got {pred.dim()}.")

        if not (target.max() < pred.size(1) and target.min() >= 0 and target.dtype == torch.long):
            raise ValueError(
                f"Expect long type target value in range (0, {pred.size(1)}). ({target.min()}, {target.max()})"
            )
        return super().forward(pred, target)

class HausdorffERLoss3D(_HausdorffERLossBase):
    r"""Binary 3D Hausdorff loss based on morphological erosion.

    Hausdorff Distance loss measures the maximum distance of a predicted segmentation boundary to
    the nearest ground-truth edge pixel. For two segmentation point sets X and Y ,
    the one-sided HD from X to Y is defined as:

    .. math::

        hd(X,Y) = \max_{x \in X} \min_{y \in Y}||x - y||_2

    Furthermore, the bidirectional HD is:

    .. math::

        HD(X,Y) = max(hd(X, Y), hd(Y, X))

    This is a 3D Hausdorff Distance (HD) Loss that based on morphological erosion, which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
    blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.

    Examples:
        >>> hdloss = HausdorffERLoss3D()
        >>> input = torch.randn(5, 3, 20, 20, 20)
        >>> target = (torch.rand(5, 1, 20, 20, 20) * 2).long()
        >>> res = hdloss(input, target)
    """

    conv = torch.conv3d
    max_pool = nn.AdaptiveMaxPool3d(1)

    def get_kernel(self) -> Tensor:
        """Get kernel for image morphology convolution."""
        cross = tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        bound = tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        # NOTE: The original repo claimed it shaped as (3, 1, 3, 3)
        #    which Jian suspect it is wrongly implemented.
        # https://github.com/PatRyg99/HausdorffLoss/blob/9f580acd421af648e74b45d46555ccb7a876c27c/hausdorff_loss.py#L94
        kernel = stack([bound, cross, bound], 1) * (1 / 7)
        return kernel[None]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute 3D Hausdorff loss.

        Args:
            pred: predicted tensor with a shape of :math:`(B, C, D, H, W)`.
                Each channel is as binary as: 1 -> fg, 0 -> bg.
            target: target tensor with a shape of :math:`(B, 1, D, H, W)`.

        Returns:
            Estimated Hausdorff Loss.
        """
        if pred.dim() != 5:
            raise ValueError(f"Only 3D images supported. Got {pred.dim()}.")

        return super().forward(pred, target)

