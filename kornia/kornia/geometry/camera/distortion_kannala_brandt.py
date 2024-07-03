def distort_points_kannala_brandt(projected_points_in_camera_z1_plane: Tensor, params: Tensor) -> Tensor:
    r"""Distort one or more points from the canonical z=1 plane into the camera frame using the Kannala-Brandt
    model.

    Args:
        projected_points_in_camera_z1_plane: Tensor representing the points to distort with shape (..., 2).
        params: Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        Tensor representing the distorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> distort_points_kannala_brandt(points, params)
        tensor([1982.6832, 1526.3619])
    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    radius_sq = x**2 + y**2

    # TODO: we can optimize this by passing the radius_sq to the impl functions. Check if it's worth it.
    distorted_points = ops.where(
        radius_sq[..., None] > 1e-8,
        _distort_points_kannala_brandt_impl(
            projected_points_in_camera_z1_plane,
            params,
        ),
        distort_points_affine(projected_points_in_camera_z1_plane, params[..., :4]),
    )

    return distorted_pointsdef undistort_points_kannala_brandt(distorted_points_in_camera: Tensor, params: Tensor) -> Tensor:
    r"""Undistort one or more points from the camera frame into the canonical z=1 plane using the Kannala-Brandt
    model.

    Args:
        distorted_points_in_camera: Tensor representing the points to undistort with shape (..., 2).
        params: Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        Tensor representing the undistorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> undistort_points_kannala_brandt(points, params).shape
        torch.Size([2])
    """
    KORNIA_CHECK_SHAPE(distorted_points_in_camera, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    x = distorted_points_in_camera[..., 0]
    y = distorted_points_in_camera[..., 1]

    fx, fy = params[..., 0], params[..., 1]
    cx, cy = params[..., 2], params[..., 3]

    k0 = params[..., 4]
    k1 = params[..., 5]
    k2 = params[..., 6]
    k3 = params[..., 7]

    un = (x - cx) / fx
    vn = (y - cy) / fy
    rth2 = un**2 + vn**2

    # TODO: explore stop condition (won't work with pytorch with batched inputs)
    # Additionally, with this stop condition we can avoid adding 1e-8 to the denominator
    # in the return statement of the function.

    # if rth2.abs() < 1e-8:
    #     return distorted_points_in_camera

    rth = rth2.sqrt()

    th = rth.sqrt()

    iters = 0

    # gauss-newton

    while True:
        th2 = th**2
        th4 = th2**2
        th6 = th2 * th4
        th8 = th4**2

        thd = th * (1.0 + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8)

        d_thd_wtr_th = 1.0 + 3.0 * k0 * th2 + 5.0 * k1 * th4 + 7.0 * k2 * th6 + 9.0 * k3 * th8

        step = (thd - rth) / d_thd_wtr_th
        th = th - step

        iters += 1

        # TODO: improve stop condition by masking only the elements that have converged
        th_abs_mask = th.abs() < 1e-8

        if th_abs_mask.all():
            break

        if iters >= 20:
            break

    radius_undistorted = th.tan()

    undistorted_points = ops.where(
        radius_undistorted[..., None] < 0.0,
        ops.stack(
            [
                -radius_undistorted * un / (rth + 1e-8),
                -radius_undistorted * vn / (rth + 1e-8),
            ],
            dim=-1,
        ),
        ops.stack(
            [
                radius_undistorted * un / (rth + 1e-8),
                radius_undistorted * vn / (rth + 1e-8),
            ],
            dim=-1,
        ),
    )

    return undistorted_points