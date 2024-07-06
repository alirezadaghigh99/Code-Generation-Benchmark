def depth_to_3d(depth: Tensor, camera_matrix: Tensor, normalize_points: bool = False) -> Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    .. note::

        This is an alternative implementation of `depth_to_3d` that does not require the creation of a meshgrid.
        In future, we will support only this implementation.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
          represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a 3d point per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_3d(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(depth, Tensor):
        raise TypeError(f"Input depht type is not a Tensor. Got {type(depth)}.")

    if not (len(depth.shape) == 4 and depth.shape[-3] == 1):
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, Tensor):
        raise TypeError(f"Input camera_matrix type is not a Tensor. Got {type(camera_matrix)}.")

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). Got: {camera_matrix.shape}.")

    # create base coordinates grid
    _, _, height, width = depth.shape
    points_2d: Tensor = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth: Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points
    )  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW

def depth_to_normals(depth: Tensor, camera_matrix: Tensor, normalize_points: bool = False) -> Tensor:
    """Compute the normal surface per pixel.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalize the pointcloud. This must be set to `True` when the depth is
        represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_normals(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(depth, Tensor):
        raise TypeError(f"Input depht type is not a Tensor. Got {type(depth)}.")

    if not (len(depth.shape) == 4 and depth.shape[-3] == 1):
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, Tensor):
        raise TypeError(f"Input camera_matrix type is not a Tensor. Got {type(camera_matrix)}.")

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). Got: {camera_matrix.shape}.")

    # compute the 3d points from depth
    xyz: Tensor = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients: Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    normals: Tensor = torch.cross(a, b, dim=1)  # Bx3xHxW
    return kornia_ops.normalize(normals, dim=1, p=2)

