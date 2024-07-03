def get_projective_transform(center: Tensor, angles: Tensor, scales: Tensor) -> Tensor:
    r"""Calculate the projection matrix for a 3D rotation.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    The function computes the projection matrix given the center and angles per axis.

    Args:
        center: center of the rotation (x,y,z) in the source with shape :math:`(B, 3)`.
        angles: axis angle vector containing the rotation angles in degrees in the form
            of (rx, ry, rz) with shape :math:`(B, 3)`. Internally it calls Rodrigues to compute
            the rotation matrix from axis-angle.
        scales: scale factor for x-y-z-directions with shape :math:`(B, 3)`.

    Returns:
        the projection matrix of 3D rotation with shape :math:`(B, 3, 4)`.

    .. note::
        This function is often used in conjunction with :func:`warp_affine3d`.
    """
    if not (len(center.shape) == 2 and center.shape[-1] == 3):
        raise AssertionError(center.shape)
    if not (len(angles.shape) == 2 and angles.shape[-1] == 3):
        raise AssertionError(angles.shape)
    if center.device != angles.device:
        raise AssertionError(center.device, angles.device)
    if center.dtype != angles.dtype:
        raise AssertionError(center.dtype, angles.dtype)

    # create rotation matrix
    axis_angle_rad: Tensor = deg2rad(angles)
    rmat: Tensor = axis_angle_to_rotation_matrix(axis_angle_rad)  # Bx3x3
    scaling_matrix: Tensor = eye_like(3, rmat)
    scaling_matrix = scaling_matrix * scales.unsqueeze(dim=1)
    rmat = rmat @ scaling_matrix.to(rmat)

    # define matrix to move forth and back to origin
    from_origin_mat = eye_like(4, rmat, shared_memory=False)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    to_origin_mat = from_origin_mat.clone()
    to_origin_mat = _torch_inverse_cast(from_origin_mat)

    # append translation with zeros
    proj_mat = projection_from_Rt(rmat, torch.zeros_like(center)[..., None])  # Bx3x4

    # chain 4x4 transforms
    proj_mat = convert_affinematrix_to_homography3d(proj_mat)  # Bx4x4
    proj_mat = from_origin_mat @ proj_mat @ to_origin_mat

    return proj_mat[..., :3, :]  # Bx3x4def get_rotation_matrix2d(center: Tensor, angle: Tensor, scale: Tensor) -> Tensor:
    r"""Calculate an affine matrix of 2D rotation.

    The function calculates the following matrix:

    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}

    where

    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})

    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.

    Args:
        center: center of the rotation in the source image with shape :math:`(B, 2)`.
        angle: rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner) with shape :math:`(B)`.
        scale: scale factor for x, y scaling with shape :math:`(B, 2)`.

    Returns:
        the affine matrix of 2D rotation with shape :math:`(B, 2, 3)`.

    Example:
        >>> center = zeros(1, 2)
        >>> scale = torch.ones((1, 2))
        >>> angle = 45. * torch.ones(1)
        >>> get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])

    .. note::
        This function is often used in conjunction with :func:`warp_affine`.
    """
    if not isinstance(center, Tensor):
        raise TypeError(f"Input center type is not a Tensor. Got {type(center)}")

    if not isinstance(angle, Tensor):
        raise TypeError(f"Input angle type is not a Tensor. Got {type(angle)}")

    if not isinstance(scale, Tensor):
        raise TypeError(f"Input scale type is not a Tensor. Got {type(scale)}")

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError(f"Input center must be a Bx2 tensor. Got {center.shape}")

    if not len(angle.shape) == 1:
        raise ValueError(f"Input angle must be a B tensor. Got {angle.shape}")

    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError(f"Input scale must be a Bx2 tensor. Got {scale.shape}")

    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError(
            f"Inputs must have same batch size dimension. Got center {center.shape}, angle {angle.shape} and scale "
            f"{scale.shape}"
        )

    if not (center.device == angle.device == scale.device) or not (center.dtype == angle.dtype == scale.dtype):
        raise ValueError(
            f"Inputs must have same device Got center ({center.device}, {center.dtype}), angle ({angle.device}, "
            f"{angle.dtype}) and scale ({scale.device}, {scale.dtype})"
        )

    shift_m = eye_like(3, center)
    shift_m[:, :2, 2] = center

    shift_m_inv = eye_like(3, center)
    shift_m_inv[:, :2, 2] = -center

    scale_m = eye_like(3, center)
    scale_m[:, 0, 0] *= scale[:, 0]
    scale_m[:, 1, 1] *= scale[:, 1]

    rotat_m = eye_like(3, center)
    rotat_m[:, :2, :2] = angle_to_rotation_matrix(angle)

    affine_m = shift_m @ rotat_m @ scale_m @ shift_m_inv
    return affine_m[:, :2, :]  # Bx2x3