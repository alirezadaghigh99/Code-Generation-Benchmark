class NDCMultinomialRaysampler(MultinomialRaysampler):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined minimum and maximum depth.

    `NDCMultinomialRaysampler` follows the screen conventions of the `Meshes` and `Pointclouds`
    renderers. I.e. the pixel coordinates are in [-1, 1]x[-u, u] or [-u, u]x[-1, 1]
    where u > 1 is the aspect ratio of the image.

    For the description of arguments, see the documentation to MultinomialRaysampler.
    """

    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: Optional[int] = None,
        n_rays_total: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
    ) -> None:
        if image_width >= image_height:
            range_x = image_width / image_height
            range_y = 1.0
        else:
            range_x = 1.0
            range_y = image_height / image_width

        half_pix_width = range_x / image_width
        half_pix_height = range_y / image_height
        super().__init__(
            min_x=range_x - half_pix_width,
            max_x=-range_x + half_pix_width,
            min_y=range_y - half_pix_height,
            max_y=-range_y + half_pix_height,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
            n_rays_per_image=n_rays_per_image,
            n_rays_total=n_rays_total,
            unit_directions=unit_directions,
            stratified_sampling=stratified_sampling,
        )

