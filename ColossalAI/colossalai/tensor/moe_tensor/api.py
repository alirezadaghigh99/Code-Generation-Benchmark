def is_moe_tensor(tensor: torch.Tensor) -> bool:
    """
    Check whether the given tensor is a moe tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        bool: Whether the given tensor is a moe tensor.
    """
    return hasattr(tensor, "ep_group")

