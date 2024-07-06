def quaternion_to_axis_angle(quaternion: Tensor) -> Tensor:
    """Convert quaternion vector to axis angle of rotation in radians.

    The quaternion should be in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion: tensor with quaternions.

    Return:
        tensor with axis angle of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = tensor((1., 0., 0., 0.))
        >>> quaternion_to_axis_angle(quaternion)
        tensor([0., 0., 0.])
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}")

    # unpack input and compute conversion
    q1: Tensor = tensor([])
    q2: Tensor = tensor([])
    q3: Tensor = tensor([])
    cos_theta: Tensor = tensor([])

    cos_theta = quaternion[..., 0]
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]

    sin_squared_theta: Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: Tensor = torch.sqrt(sin_squared_theta)
    two_theta: Tensor = 2.0 * where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: Tensor = two_theta / sin_theta
    k_neg: Tensor = 2.0 * torch.ones_like(sin_theta)
    k: Tensor = where(sin_squared_theta > 0.0, k_pos, k_neg)

    axis_angle: Tensor = torch.zeros_like(quaternion)[..., :3]
    axis_angle[..., 0] += q1 * k
    axis_angle[..., 1] += q2 * k
    axis_angle[..., 2] += q3 * k
    return axis_angle

def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    r"""Convert a quaternion to a rotation matrix.

    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.

    Return:
        the rotation matrix of shape :math:`(*, 3, 3)`.

    Example:
        >>> quaternion = tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # normalize the input quaternion
    quaternion_norm: Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    w = quaternion_norm[..., 0]
    x = quaternion_norm[..., 1]
    y = quaternion_norm[..., 2]
    z = quaternion_norm[..., 3]

    # compute the actual conversion
    tx: Tensor = 2.0 * x
    ty: Tensor = 2.0 * y
    tz: Tensor = 2.0 * z
    twx: Tensor = tx * w
    twy: Tensor = ty * w
    twz: Tensor = tz * w
    txx: Tensor = tx * x
    txy: Tensor = ty * x
    txz: Tensor = tz * x
    tyy: Tensor = ty * y
    tyz: Tensor = tz * y
    tzz: Tensor = tz * z
    one: Tensor = tensor(1.0)

    matrix_flat: Tensor = stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    )

    # this slightly awkward construction of the output shape is to satisfy torchscript
    output_shape = [*list(quaternion.shape[:-1]), 3, 3]
    matrix = matrix_flat.reshape(output_shape)

    return matrix

def quaternion_exp_to_log(quaternion: Tensor, eps: float = 1.0e-8) -> Tensor:
    r"""Apply the log map to a quaternion.

    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.
        eps: a small number for clamping.

    Return:
        the quaternion log map of shape :math:`(*, 3)`.

    Example:
        >>> quaternion = tensor((1., 0., 0., 0.))
        >>> quaternion_exp_to_log(quaternion, eps=torch.finfo(quaternion.dtype).eps)
        tensor([0., 0., 0.])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # unpack quaternion vector and scalar
    quaternion_vector: Tensor = tensor([])
    quaternion_scalar: Tensor = tensor([])

    quaternion_scalar = quaternion[..., 0:1]
    quaternion_vector = quaternion[..., 1:4]

    # compute quaternion norm
    norm_q: Tensor = torch.norm(quaternion_vector, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # apply log map
    quaternion_log: Tensor = quaternion_vector * torch.acos(torch.clamp(quaternion_scalar, min=-1.0, max=1.0)) / norm_q

    return quaternion_log

def rotation_matrix_to_quaternion(rotation_matrix: Tensor, eps: float = 1.0e-8) -> Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) format.

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(*, 3, 3)`.
        eps: small value to avoid zero division.

    Return:
        the rotation in quaternion with shape :math:`(*, 4)`.

    Example:
        >>> input = tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(rotation_matrix, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    def safe_zero_division(numerator: Tensor, denominator: Tensor) -> Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: Tensor = rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: Tensor = m00 + m11 + m22

    def trace_positive_cond() -> Tensor:
        sq = torch.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return concatenate((qw, qx, qy, qz), dim=-1)

    def cond_1() -> Tensor:
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return concatenate((qw, qx, qy, qz), dim=-1)

    def cond_2() -> Tensor:
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return concatenate((qw, qx, qy, qz), dim=-1)

    def cond_3() -> Tensor:
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return concatenate((qw, qx, qy, qz), dim=-1)

    where_2 = where(m11 > m22, cond_2(), cond_3())
    where_1 = where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: Tensor = where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion

def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors

    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3

def normal_transform_pixel3d(
    depth: int,
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        depth: image depth.
        height: image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors

    Returns:
        normalized transform with shape :math:`(1, 4, 4)`.
    """
    tr_mat = tensor(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 4x4

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0
    depth_denom: float = eps if depth == 1 else depth - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    tr_mat[2, 2] = tr_mat[2, 2] * 2.0 / depth_denom

    return tr_mat.unsqueeze(0)  # 1x4x4

def convert_points_from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

    Examples:
        >>> input = tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # we check for points at max_val
    z_vec: Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: Tensor = torch.abs(z_vec) > eps
    scale = where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]

