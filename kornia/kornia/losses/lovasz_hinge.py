class LovaszHingeLoss(nn.Module):
    r"""Criterion that computes a surrogate binary intersection-over-union (IoU) loss.

    According to [2], we compute the IoU as follows:

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
        [1] http://proceedings.mlr.press/v37/yub15.pdf
        [2] https://arxiv.org/pdf/1705.08790.pdf

    .. note::
        This loss function only supports binary labels. For multi-class labels please
        use the Lovasz-Softmax loss.

    Args:
        pred: logits tensor with shape :math:`(N, 1, H, W)`.
        labels: labels tensor with shape :math:`(N, H, W)` with binary values.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 1  # num_classes
        >>> criterion = LovaszHingeLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return lovasz_hinge_loss(pred=pred, target=target)

