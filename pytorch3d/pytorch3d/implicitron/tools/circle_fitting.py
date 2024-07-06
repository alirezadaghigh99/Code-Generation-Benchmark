def fit_circle_in_3d(
    points,
    *,
    n_points: int = 0,
    angles: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    up: Optional[torch.Tensor] = None,
) -> Circle3D:
    """
    Simple best fit circle to 3D points. Uses circle_2d in the
    least-squares best fit plane.

    In addition, generates points along the circle. If angles is None (default)
    then n_points around the circle equally spaced are given. These begin at the
    point closest to the first input point. They continue in the direction which
    seems to be match the movement of points. If angles is provided, then n_points
    is ignored, and points along the circle at the given angles are returned,
    with the starting point and direction as before.

    Further, an offset can be given to add to the generated points; this is
    interpreted in a rotated coordinate system where (0, 0, 1) is normal to the
    circle, specifically the normal which is approximately in the direction of a
    given `up` vector. The remaining rotation is disambiguated in an unspecified
    but deterministic way.

    (Note that `generated_points` is affected by the order of the points in
    points, but the other outputs are not.)

    Args:
        points2d: N x 3 tensor of 3D points
        n_points: number of points to generate on the circle
        angles: optional angles in radians of points to generate.
        offset: optional tensor (3,), a displacement expressed in a "canonical"
                coordinate system to add to the generated points.
        up: optional tensor (3,), a vector which helps define the
            "canonical" coordinate system for interpretting `offset`.
            Required if offset is used.


    Returns:
        Circle3D object
    """
    centroid = points.mean(0)
    r = get_rotation_to_best_fit_xy(points, centroid)
    normal = r[:, 2]
    rotated_points = (points - centroid) @ r
    result_2d = fit_circle_in_2d(
        rotated_points[:, :2], n_points=n_points, angles=angles
    )
    center_3d = result_2d.center @ r[:, :2].t() + centroid
    n_generated_points = result_2d.generated_points.shape[0]
    if n_generated_points > 0:
        generated_points_in_plane = torch.cat(
            [
                result_2d.generated_points,
                torch.zeros_like(result_2d.generated_points[:, :1]),
            ],
            dim=1,
        )
        if offset is not None:
            if up is None:
                raise ValueError("Missing `up` input for interpreting offset")
            with torch.no_grad():
                swap = torch.dot(up, normal) < 0
            if swap:
                # We need some rotation which takes +z to -z. Here's one.
                generated_points_in_plane += offset * offset.new_tensor([1, -1, -1])
            else:
                generated_points_in_plane += offset

        generated_points = generated_points_in_plane @ r.t() + centroid
    else:
        generated_points = points.new_zeros(0, 3)

    return Circle3D(
        radius=result_2d.radius,
        center=center_3d,
        normal=normal,
        generated_points=generated_points,
    )

