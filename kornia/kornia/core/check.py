def KORNIA_CHECK_LAF(laf: Tensor, raises: bool = True) -> bool:
    """Check whether a Local Affine Frame (laf) has a valid shape.

    Args:
        laf: local affine frame tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the input laf does not have a shape :math:`(B,N,2,3)` and raises is True.

    Example:
        >>> lafs = torch.rand(2, 10, 2, 3)
        >>> KORNIA_CHECK_LAF(lafs)
        True
    """
    return KORNIA_CHECK_SHAPE(laf, ["B", "N", "2", "3"], raises)

