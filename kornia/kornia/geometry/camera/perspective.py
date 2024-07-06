def project_points(point_3d: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    r"""Project a 3d point onto the 2d camera plane.

    Args:
        point3d: tensor containing the 3d points to be projected
            to the camera plane. The shape of the tensor can be :math:`(*, 3)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> X = torch.rand(1, 3)
        >>> K = torch.eye(3)[None]
        >>> project_points(X, K)
        tensor([[5.6088, 8.6827]])
    """
    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X / Z + cx
    # v = fy * Y / Z + cy
    # project back using depth dividing in a safe way
    xy_coords: torch.Tensor = convert_points_from_homogeneous(point_3d)
    return denormalize_points_with_intrinsics(xy_coords, camera_matrix)

