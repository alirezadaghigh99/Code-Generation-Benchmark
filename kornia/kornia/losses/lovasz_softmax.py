class LovaszSoftmaxLoss(nn.Module):
    r"""Criterion that computes a surrogate multi-class intersection-over-union (IoU) loss.

    According to [1], we compute the IoU as follows:

    .. math::

        \text{IoU}(x, class) = \frac{|X \cap Y|}{|X \cup Y|}

    [1] approximates this fomular with a surrogate, which is fully differentable.

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the binary tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{IoU}(x, class)

    Reference:
        [1] https://arxiv.org/pdf/1705.08790.pdf

    .. note::
        This loss function only supports multi-class (C > 1) labels. For binary
        labels please use the Lovasz-Hinge loss.

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = LovaszSoftmaxLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(self, weight: Optional[Tensor] = None) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return lovasz_softmax_loss(pred=pred, target=target, weight=self.weight)

