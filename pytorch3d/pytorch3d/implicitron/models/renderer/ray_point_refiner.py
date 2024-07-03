def apply_blurpool_on_weights(weights) -> torch.Tensor:
    """
    Filter weights with a 2-tap max filters followed by a 2-tap blur filter,
    which produces a wide and smooth upper envelope on the weights.

    Args:
        weights: Tensor of shape `(..., dim)`

    Returns:
        blured_weights: Tensor of shape `(..., dim)`
    """
    weights_pad = torch.concatenate(
        [
            weights[..., :1],
            weights,
            weights[..., -1:],
        ],
        dim=-1,
    )

    weights_max = torch.nn.functional.max_pool1d(
        weights_pad.flatten(end_dim=-2), 2, stride=1
    )
    return torch.lerp(weights_max[..., :-1], weights_max[..., 1:], 0.5).reshape_as(
        weights
    )