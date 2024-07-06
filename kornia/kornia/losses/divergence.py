def kl_div_loss_2d(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    r"""Calculate the Kullback-Leibler divergence loss between heatmaps.

    Args:
        pred: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> pred = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = kl_div_loss_2d(pred, pred)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(_kl_div_2d(target, pred), reduction)

def js_div_loss_2d(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    r"""Calculate the Jensen-Shannon divergence loss between heatmaps.

    Args:
        pred: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> pred = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(pred, pred)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(_js_div_2d(target, pred), reduction)

