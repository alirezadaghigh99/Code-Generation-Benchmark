class NearFarRaySampler(AbstractMaskRaySampler):
    """
    Samples a fixed number of points between fixed near and far z-planes.
    Specifically, samples points along each ray with approximately uniform spacing
    of z-coordinates between the minimum depth `self.min_depth` and the maximum depth
    `self.max_depth`. This sampling is useful for rendering scenes where the camera is
    in a constant distance from the focal point of the scene.

    Args:
        min_depth: The minimum depth of a ray-point.
        max_depth: The maximum depth of a ray-point.
    """

    min_depth: float = 0.1
    max_depth: float = 8.0

    def _get_min_max_depth_bounds(self, cameras: CamerasBase) -> Tuple[float, float]:
        """
        Returns the stored near/far planes.
        """
        return self.min_depth, self.max_depth

class AdaptiveRaySampler(AbstractMaskRaySampler):
    """
    Adaptively samples points on each ray between near and far planes whose
    depths are determined based on the distance from the camera center
    to a predefined scene center.

    More specifically,
    `min_depth = max(
        (self.scene_center-camera_center).norm() - self.scene_extent, eps
    )` and
    `max_depth = (self.scene_center-camera_center).norm() + self.scene_extent`.

    This sampling is ideal for object-centric scenes whose contents are
    centered around a known `self.scene_center` and fit into a bounding sphere
    with a radius of `self.scene_extent`.

    Args:
        scene_center: The xyz coordinates of the center of the scene used
            along with `scene_extent` to compute the min and max depth planes
            for sampling ray-points.
        scene_extent: The radius of the scene bounding box centered at `scene_center`.
    """

    scene_extent: float = 8.0
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()
        if self.scene_extent <= 0.0:
            raise ValueError("Adaptive raysampler requires self.scene_extent > 0.")
        self._scene_center = torch.FloatTensor(self.scene_center)

    def _get_min_max_depth_bounds(self, cameras: CamerasBase) -> Tuple[float, float]:
        """
        Returns the adaptively calculated near/far planes.
        """
        min_depth, max_depth = camera_utils.get_min_max_depth_bounds(
            cameras, self._scene_center, self.scene_extent
        )
        return float(min_depth[0]), float(max_depth[0])

