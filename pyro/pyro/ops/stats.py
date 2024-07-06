def quantile(input, probs, dim=0):
    """
    Computes quantiles of ``input`` at ``probs``. If ``probs`` is a scalar,
    the output will be squeezed at ``dim``.

    :param torch.Tensor input: the input tensor.
    :param list probs: quantile positions.
    :param int dim: dimension to take quantiles from ``input``.
    :returns torch.Tensor: quantiles of ``input`` at ``probs``.
    """
    if isinstance(probs, (numbers.Number, list, tuple)):
        probs = torch.tensor(probs, dtype=input.dtype, device=input.device)
    sorted_input = input.sort(dim)[0]
    max_index = input.size(dim) - 1
    indices = probs * max_index
    # because indices is float, we interpolate the quantiles linearly from nearby points
    indices_below = indices.long()
    indices_above = (indices_below + 1).clamp(max=max_index)
    quantiles_above = sorted_input.index_select(dim, indices_above)
    quantiles_below = sorted_input.index_select(dim, indices_below)
    shape_to_broadcast = [1] * input.dim()
    shape_to_broadcast[dim] = indices.numel()
    weights_above = indices - indices_below.type_as(indices)
    weights_above = weights_above.reshape(shape_to_broadcast)
    weights_below = 1 - weights_above
    quantiles = weights_below * quantiles_below + weights_above * quantiles_above
    return quantiles if probs.shape != torch.Size([]) else quantiles.squeeze(dim)

def pi(input, prob, dim=0):
    """
    Computes percentile interval which assigns equal probability mass
    to each tail of the interval.

    :param torch.Tensor input: the input tensor.
    :param float prob: the probability mass of samples within the interval.
    :param int dim: dimension to calculate percentile interval from ``input``.
    :returns torch.Tensor: quantiles of ``input`` at ``probs``.
    """
    return quantile(input, [(1 - prob) / 2, (1 + prob) / 2], dim)

