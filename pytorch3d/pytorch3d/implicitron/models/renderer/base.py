class ImplicitronRayBundle:
    """
    Parametrizes points along projection rays by storing ray `origins`,
    `directions` vectors and `lengths` at which the ray-points are sampled.
    Furthermore, the xy-locations (`xys`) of the ray pixels are stored as well.
    Note that `directions` don't have to be normalized; they define unit vectors
    in the respective 1D coordinate systems; see documentation for
    :func:`ray_bundle_to_ray_points` for the conversion formula.

    Ray bundle may represent rays from multiple cameras. In that case, cameras
    are stored in the packed form (i.e. rays from the same camera are stored in
    the consecutive elements). The following indices will be set:
        camera_ids: A tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of different
            sampled cameras.
        camera_counts: A tensor of shape (N, ) which how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts) == minibatch`, where `minibatch = origins.shape[0]`.

    Attributes:
        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
        camera_ids: An optional tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of unique sampled cameras.
        camera_counts: An optional tensor of shape (N, ) indicates how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts)==total_number_of_rays`.
        bins: An optional tensor of shape `(..., num_points_per_ray + 1)`
            containing the bins at which the rays are sampled. In this case
            lengths should be equal to the midpoints of bins `(..., num_points_per_ray)`.
        pixel_radii_2d: An optional tensor of shape `(..., 1)`
            base radii of the conical frustums.

    Raises:
        ValueError: If either bins or lengths are not provided.
        ValueError: If bins is provided and the last dim is inferior or equal to 1.
    """

    def __init__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        lengths: Optional[torch.Tensor],
        xys: torch.Tensor,
        camera_ids: Optional[torch.LongTensor] = None,
        camera_counts: Optional[torch.LongTensor] = None,
        bins: Optional[torch.Tensor] = None,
        pixel_radii_2d: Optional[torch.Tensor] = None,
    ):
        if bins is not None and bins.shape[-1] <= 1:
            raise ValueError(
                "The last dim of bins must be at least superior or equal to 2."
            )

        if bins is None and lengths is None:
            raise ValueError(
                "Please set either bins or lengths to initialize an ImplicitronRayBundle."
            )

        self.origins = origins
        self.directions = directions
        self._lengths = lengths if bins is None else None
        self.xys = xys
        self.bins = bins
        self.pixel_radii_2d = pixel_radii_2d
        self.camera_ids = camera_ids
        self.camera_counts = camera_counts

    @property
    def lengths(self) -> torch.Tensor:
        if self.bins is not None:
            # equivalent to: 0.5 * (bins[..., 1:] + bins[..., :-1]) but more efficient
            # pyre-ignore
            return torch.lerp(self.bins[..., :-1], self.bins[..., 1:], 0.5)
        return self._lengths

    @lengths.setter
    def lengths(self, value):
        if self.bins is not None:
            raise ValueError(
                "If the bins attribute is not None you cannot set the lengths attribute."
            )
        else:
            self._lengths = value

    def float_(self) -> None:
        """Moves the tensors to float dtype in place
        (helpful for mixed-precision tensors).
        """
        self.origins = self.origins.float()
        self.directions = self.directions.float()
        self._lengths = self._lengths.float() if self._lengths is not None else None
        self.xys = self.xys.float()
        self.bins = self.bins.float() if self.bins is not None else None
        self.pixel_radii_2d = (
            self.pixel_radii_2d.float() if self.pixel_radii_2d is not None else None
        )

    def is_packed(self) -> bool:
        """
        Returns whether the ImplicitronRayBundle carries data in packed state
        """
        return self.camera_ids is not None and self.camera_counts is not None

    def get_padded_xys(self) -> Tuple[torch.Tensor, torch.LongTensor, int]:
        """
        For a packed ray bundle, returns padded rays. Assumes the input bundle is packed
        (i.e. `camera_ids` and `camera_counts` are set).

        Returns:
            - xys: Tensor of shape (N, max_size, ...) containing the padded
                representation of the pixel coordinated;
                where max_size is max of `camera_counts`. The values for camera id `i`
                will be copied to `xys[i, :]`, with zeros padding out the extra inputs.
            - first_idxs: cumulative sum of `camera_counts` defininf the boundaries
                between cameras in the packed representation
            - num_inputs: the number of cameras in the bundle.
        """
        if not self.is_packed():
            raise ValueError("get_padded_xys can be called only on a packed bundle")

        camera_counts = self.camera_counts
        assert camera_counts is not None

        cumsum = torch.cumsum(camera_counts, dim=0, dtype=torch.long)
        first_idxs = torch.cat(
            (camera_counts.new_zeros((1,), dtype=torch.long), cumsum[:-1])
        )
        num_inputs = camera_counts.sum().item()
        max_size = torch.max(camera_counts).item()
        xys = packed_to_padded(self.xys, first_idxs, max_size)
        # pyre-ignore [7] pytorch typeshed inaccuracy
        return xys, first_idxs, num_inputs

class ImplicitronRayBundle:
    """
    Parametrizes points along projection rays by storing ray `origins`,
    `directions` vectors and `lengths` at which the ray-points are sampled.
    Furthermore, the xy-locations (`xys`) of the ray pixels are stored as well.
    Note that `directions` don't have to be normalized; they define unit vectors
    in the respective 1D coordinate systems; see documentation for
    :func:`ray_bundle_to_ray_points` for the conversion formula.

    Ray bundle may represent rays from multiple cameras. In that case, cameras
    are stored in the packed form (i.e. rays from the same camera are stored in
    the consecutive elements). The following indices will be set:
        camera_ids: A tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of different
            sampled cameras.
        camera_counts: A tensor of shape (N, ) which how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts) == minibatch`, where `minibatch = origins.shape[0]`.

    Attributes:
        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
        camera_ids: An optional tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of unique sampled cameras.
        camera_counts: An optional tensor of shape (N, ) indicates how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts)==total_number_of_rays`.
        bins: An optional tensor of shape `(..., num_points_per_ray + 1)`
            containing the bins at which the rays are sampled. In this case
            lengths should be equal to the midpoints of bins `(..., num_points_per_ray)`.
        pixel_radii_2d: An optional tensor of shape `(..., 1)`
            base radii of the conical frustums.

    Raises:
        ValueError: If either bins or lengths are not provided.
        ValueError: If bins is provided and the last dim is inferior or equal to 1.
    """

    def __init__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        lengths: Optional[torch.Tensor],
        xys: torch.Tensor,
        camera_ids: Optional[torch.LongTensor] = None,
        camera_counts: Optional[torch.LongTensor] = None,
        bins: Optional[torch.Tensor] = None,
        pixel_radii_2d: Optional[torch.Tensor] = None,
    ):
        if bins is not None and bins.shape[-1] <= 1:
            raise ValueError(
                "The last dim of bins must be at least superior or equal to 2."
            )

        if bins is None and lengths is None:
            raise ValueError(
                "Please set either bins or lengths to initialize an ImplicitronRayBundle."
            )

        self.origins = origins
        self.directions = directions
        self._lengths = lengths if bins is None else None
        self.xys = xys
        self.bins = bins
        self.pixel_radii_2d = pixel_radii_2d
        self.camera_ids = camera_ids
        self.camera_counts = camera_counts

    @property
    def lengths(self) -> torch.Tensor:
        if self.bins is not None:
            # equivalent to: 0.5 * (bins[..., 1:] + bins[..., :-1]) but more efficient
            # pyre-ignore
            return torch.lerp(self.bins[..., :-1], self.bins[..., 1:], 0.5)
        return self._lengths

    @lengths.setter
    def lengths(self, value):
        if self.bins is not None:
            raise ValueError(
                "If the bins attribute is not None you cannot set the lengths attribute."
            )
        else:
            self._lengths = value

    def float_(self) -> None:
        """Moves the tensors to float dtype in place
        (helpful for mixed-precision tensors).
        """
        self.origins = self.origins.float()
        self.directions = self.directions.float()
        self._lengths = self._lengths.float() if self._lengths is not None else None
        self.xys = self.xys.float()
        self.bins = self.bins.float() if self.bins is not None else None
        self.pixel_radii_2d = (
            self.pixel_radii_2d.float() if self.pixel_radii_2d is not None else None
        )

    def is_packed(self) -> bool:
        """
        Returns whether the ImplicitronRayBundle carries data in packed state
        """
        return self.camera_ids is not None and self.camera_counts is not None

    def get_padded_xys(self) -> Tuple[torch.Tensor, torch.LongTensor, int]:
        """
        For a packed ray bundle, returns padded rays. Assumes the input bundle is packed
        (i.e. `camera_ids` and `camera_counts` are set).

        Returns:
            - xys: Tensor of shape (N, max_size, ...) containing the padded
                representation of the pixel coordinated;
                where max_size is max of `camera_counts`. The values for camera id `i`
                will be copied to `xys[i, :]`, with zeros padding out the extra inputs.
            - first_idxs: cumulative sum of `camera_counts` defininf the boundaries
                between cameras in the packed representation
            - num_inputs: the number of cameras in the bundle.
        """
        if not self.is_packed():
            raise ValueError("get_padded_xys can be called only on a packed bundle")

        camera_counts = self.camera_counts
        assert camera_counts is not None

        cumsum = torch.cumsum(camera_counts, dim=0, dtype=torch.long)
        first_idxs = torch.cat(
            (camera_counts.new_zeros((1,), dtype=torch.long), cumsum[:-1])
        )
        num_inputs = camera_counts.sum().item()
        max_size = torch.max(camera_counts).item()
        xys = packed_to_padded(self.xys, first_idxs, max_size)
        # pyre-ignore [7] pytorch typeshed inaccuracy
        return xys, first_idxs, num_inputs

class ImplicitronRayBundle:
    """
    Parametrizes points along projection rays by storing ray `origins`,
    `directions` vectors and `lengths` at which the ray-points are sampled.
    Furthermore, the xy-locations (`xys`) of the ray pixels are stored as well.
    Note that `directions` don't have to be normalized; they define unit vectors
    in the respective 1D coordinate systems; see documentation for
    :func:`ray_bundle_to_ray_points` for the conversion formula.

    Ray bundle may represent rays from multiple cameras. In that case, cameras
    are stored in the packed form (i.e. rays from the same camera are stored in
    the consecutive elements). The following indices will be set:
        camera_ids: A tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of different
            sampled cameras.
        camera_counts: A tensor of shape (N, ) which how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts) == minibatch`, where `minibatch = origins.shape[0]`.

    Attributes:
        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
        camera_ids: An optional tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of unique sampled cameras.
        camera_counts: An optional tensor of shape (N, ) indicates how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts)==total_number_of_rays`.
        bins: An optional tensor of shape `(..., num_points_per_ray + 1)`
            containing the bins at which the rays are sampled. In this case
            lengths should be equal to the midpoints of bins `(..., num_points_per_ray)`.
        pixel_radii_2d: An optional tensor of shape `(..., 1)`
            base radii of the conical frustums.

    Raises:
        ValueError: If either bins or lengths are not provided.
        ValueError: If bins is provided and the last dim is inferior or equal to 1.
    """

    def __init__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        lengths: Optional[torch.Tensor],
        xys: torch.Tensor,
        camera_ids: Optional[torch.LongTensor] = None,
        camera_counts: Optional[torch.LongTensor] = None,
        bins: Optional[torch.Tensor] = None,
        pixel_radii_2d: Optional[torch.Tensor] = None,
    ):
        if bins is not None and bins.shape[-1] <= 1:
            raise ValueError(
                "The last dim of bins must be at least superior or equal to 2."
            )

        if bins is None and lengths is None:
            raise ValueError(
                "Please set either bins or lengths to initialize an ImplicitronRayBundle."
            )

        self.origins = origins
        self.directions = directions
        self._lengths = lengths if bins is None else None
        self.xys = xys
        self.bins = bins
        self.pixel_radii_2d = pixel_radii_2d
        self.camera_ids = camera_ids
        self.camera_counts = camera_counts

    @property
    def lengths(self) -> torch.Tensor:
        if self.bins is not None:
            # equivalent to: 0.5 * (bins[..., 1:] + bins[..., :-1]) but more efficient
            # pyre-ignore
            return torch.lerp(self.bins[..., :-1], self.bins[..., 1:], 0.5)
        return self._lengths

    @lengths.setter
    def lengths(self, value):
        if self.bins is not None:
            raise ValueError(
                "If the bins attribute is not None you cannot set the lengths attribute."
            )
        else:
            self._lengths = value

    def float_(self) -> None:
        """Moves the tensors to float dtype in place
        (helpful for mixed-precision tensors).
        """
        self.origins = self.origins.float()
        self.directions = self.directions.float()
        self._lengths = self._lengths.float() if self._lengths is not None else None
        self.xys = self.xys.float()
        self.bins = self.bins.float() if self.bins is not None else None
        self.pixel_radii_2d = (
            self.pixel_radii_2d.float() if self.pixel_radii_2d is not None else None
        )

    def is_packed(self) -> bool:
        """
        Returns whether the ImplicitronRayBundle carries data in packed state
        """
        return self.camera_ids is not None and self.camera_counts is not None

    def get_padded_xys(self) -> Tuple[torch.Tensor, torch.LongTensor, int]:
        """
        For a packed ray bundle, returns padded rays. Assumes the input bundle is packed
        (i.e. `camera_ids` and `camera_counts` are set).

        Returns:
            - xys: Tensor of shape (N, max_size, ...) containing the padded
                representation of the pixel coordinated;
                where max_size is max of `camera_counts`. The values for camera id `i`
                will be copied to `xys[i, :]`, with zeros padding out the extra inputs.
            - first_idxs: cumulative sum of `camera_counts` defininf the boundaries
                between cameras in the packed representation
            - num_inputs: the number of cameras in the bundle.
        """
        if not self.is_packed():
            raise ValueError("get_padded_xys can be called only on a packed bundle")

        camera_counts = self.camera_counts
        assert camera_counts is not None

        cumsum = torch.cumsum(camera_counts, dim=0, dtype=torch.long)
        first_idxs = torch.cat(
            (camera_counts.new_zeros((1,), dtype=torch.long), cumsum[:-1])
        )
        num_inputs = camera_counts.sum().item()
        max_size = torch.max(camera_counts).item()
        xys = packed_to_padded(self.xys, first_idxs, max_size)
        # pyre-ignore [7] pytorch typeshed inaccuracy
        return xys, first_idxs, num_inputs

