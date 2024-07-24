class HeterogeneousRayBundle:
    """
    Members:
        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
        camera_ids: A tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of unique sampled cameras.
        camera_counts: A tensor of shape (N, ) which how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts)==total_number_of_rays`

    If we sample cameras of ids [0, 3, 5, 3, 1, 0, 0] that would be
    stored as camera_ids=[1, 3, 5, 0] and camera_counts=[1, 2, 1, 3]. `camera_ids` is a
    set like object with no particular ordering of elements. ith element of
    `camera_ids` coresponds to the ith element of `camera_counts`.
    """

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor
    camera_ids: Optional[torch.LongTensor] = None
    camera_counts: Optional[torch.LongTensor] = None

