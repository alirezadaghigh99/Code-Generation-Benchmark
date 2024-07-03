def find_fundamental(
    points1: Tensor, points2: Tensor, weights: Optional[Tensor] = None, method: Literal["8POINT", "7POINT"] = "8POINT"
) -> Tensor:
    r"""
    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        method: The method to use for computing the fundamental matrix. Supported methods are "7POINT" and "8POINT".

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3)`, where `m` number of fundamental matrix.

    Raises:
        ValueError: If an invalid method is provided.

    """
    if method.upper() == "7POINT":
        result = run_7point(points1, points2)
    elif method.upper() == "8POINT":
        result = run_8point(points1, points2, weights)
    else:
        raise ValueError(f"Invalid method: {method}. Supported methods are '7POINT' and '8POINT'.")
    return resultdef normalize_transformation(M: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Normalize a given transformation matrix.

    The function trakes the transformation matrix and normalize so that the value in
    the last row and column is one.

    Args:
        M: The transformation to be normalized of any shape with a minimum size of 2x2.
        eps: small value to avoid unstabilities during the backpropagation.

    Returns:
        the normalized transformation matrix with same shape as the input.
    """
    if len(M.shape) < 2:
        raise AssertionError(M.shape)
    norm_val: Tensor = M[..., -1:, -1:]
    return where(norm_val.abs() > eps, M / (norm_val + eps), M)