def distribute_points(geometries: Union[tuple, list, Geometry, float],
                      dim: Shape = instance('points'),
                      points_per_cell: int = 8,
                      center: bool = False,
                      radius: float = None,
                      extrapolation: Union[float, Extrapolation] = math.NAN,
                      **domain) -> PointCloud:
    """
    Transforms `Geometry` objects into a PointCloud.

    Args:
        geometries: Geometry objects marking the cells which should contain points
        dim: Dimension along which the points are listed.
        points_per_cell: Number of points for each cell of `geometries`
        center: Set all points to the center of the grid cells.
        radius: Sphere radius.
        extrapolation: Extrapolation for the `PointCloud`, default `NaN` used for FLIP.

    Returns:
         PointCloud representation of `geometries`.
    """
    warnings.warn("distribute_points() is deprecated. Construct a PointCloud directly.", DeprecationWarning)
    from phi.field import CenteredGrid
    if isinstance(geometries, (tuple, list, Geometry)):
        from phi.geom import union
        geometries = union(geometries)
    geometries = resample(geometries, CenteredGrid(0, extrapolation, **domain), scatter=False)
    initial_points = _distribute_points(geometries.values, dim, points_per_cell, center=center)
    if radius is None:
        from phi.field._field_math import data_bounds
        radius = math.mean(data_bounds(initial_points).size) * 0.005
    from phi.geom import Sphere
    return PointCloud(Sphere(initial_points, radius=radius), extrapolation=geometries.extrapolation, bounds=geometries.bounds)

