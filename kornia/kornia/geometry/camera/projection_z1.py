def unproject_points_z1(points_in_cam_canonical: Tensor, extension: Optional[Tensor] = None) -> Tensor:
    r"""Unproject one or more points from the canonical z=1 plane into the camera frame.

    .. math::
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} =
        \begin{bmatrix} u \\ v \end{bmatrix} \cdot w

    Args:
        points_in_cam_canonical: Tensor representing the points to unproject with shape (..., 2).
        extension: Tensor representing the extension (depth) of the points to unproject with shape (..., 1).

    Returns:
        Tensor representing the unprojected points with shape (..., 3).

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> extension = torch.tensor([3.])
        >>> unproject_points_z1(points, extension)
        tensor([3., 6., 3.])
    """
    KORNIA_CHECK_SHAPE(points_in_cam_canonical, ["*", "2"])

    if extension is None:
        extension = ops.ones(
            points_in_cam_canonical.shape[:-1] + (1,),
            device=points_in_cam_canonical.device,
            dtype=points_in_cam_canonical.dtype,
        )  # (..., 1)
    elif extension.shape[0] > 1:
        extension = extension[..., None]  # (..., 1)

    return ops.concatenate([points_in_cam_canonical * extension, extension], dim=-1)

def project_points_z1(points_in_camera: Tensor) -> Tensor:
    r"""Project one or more points from the camera frame into the canonical z=1 plane through perspective division.

    .. math::

        \begin{bmatrix} u \\ v \\ w \end{bmatrix} =
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} / z

    .. note::

        This function has a precondition that the points are in front of the camera, i.e. z > 0.
        If this is not the case, the points will be projected to the canonical plane, but the resulting
        points will be behind the camera and causing numerical issues for z == 0.

    Args:
        points_in_camera: Tensor representing the points to project with shape (..., 3).

    Returns:
        Tensor representing the projected points with shape (..., 2).

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> project_points_z1(points)
        tensor([0.3333, 0.6667])
    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    return points_in_camera[..., :2] / points_in_camera[..., 2:3]

