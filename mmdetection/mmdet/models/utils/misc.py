def reweight_loss_dict(losses: dict, weight: float) -> dict:
    """Reweight losses in the dict by weight.

    Args:
        losses (dict):  A dictionary of loss components.
        weight (float): Weight for loss components.

    Returns:
            dict: A dictionary of weighted loss components.
    """
    for name, loss in losses.items():
        if 'loss' in name:
            if isinstance(loss, Sequence):
                losses[name] = [item * weight for item in loss]
            else:
                losses[name] = loss * weight
    return losses