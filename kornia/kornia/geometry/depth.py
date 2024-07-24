def depth_to_3d(depth: Tensor, camera_matrix: Tensor, normalize_points: bool = False) -> Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    .. note::

        This is an alternative implementation of `depth_to_3d` that does not require the creation of a meshgrid.
        In future, we will support only this implementation.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
          represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a 3d point per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_3d(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(depth, Tensor):
        raise TypeError(f"Input depht type is not a Tensor. Got {type(depth)}.")

    if not (len(depth.shape) == 4 and depth.shape[-3] == 1):
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, Tensor):
        raise TypeError(f"Input camera_matrix type is not a Tensor. Got {type(camera_matrix)}.")

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). Got: {camera_matrix.shape}.")

    # create base coordinates grid
    _, _, height, width = depth.shape
    points_2d: Tensor = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth: Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points
    )  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW

def depth_to_normals(depth: Tensor, camera_matrix: Tensor, normalize_points: bool = False) -> Tensor:
    """Compute the normal surface per pixel.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalize the pointcloud. This must be set to `True` when the depth is
        represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_normals(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(depth, Tensor):
        raise TypeError(f"Input depht type is not a Tensor. Got {type(depth)}.")

    if not (len(depth.shape) == 4 and depth.shape[-3] == 1):
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, Tensor):
        raise TypeError(f"Input camera_matrix type is not a Tensor. Got {type(camera_matrix)}.")

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). Got: {camera_matrix.shape}.")

    # compute the 3d points from depth
    xyz: Tensor = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients: Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    normals: Tensor = torch.cross(a, b, dim=1)  # Bx3xHxW
    return kornia_ops.normalize(normals, dim=1, p=2)

class DepthWarper(Module):
    r"""Warp a patch by depth.

    .. math::
        P_{src}^{\{dst\}} = K_{dst} * T_{src}^{\{dst\}}

        I_{src} = \\omega(I_{dst}, P_{src}^{\{dst\}}, D_{src})

    Args:
        pinholes_dst: the pinhole models for the destination frame.
        height: the height of the image to warp.
        width: the width of the image to warp.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
    """

    def __init__(
        self,
        pinhole_dst: PinholeCamera,
        height: int,
        width: int,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        # constructor members
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.eps = 1e-6
        self.align_corners: bool = align_corners

        # state members
        self._pinhole_dst: PinholeCamera = pinhole_dst
        self._pinhole_src: None | PinholeCamera = None
        self._dst_proj_src: None | Tensor = None

        self.grid: Tensor = self._create_meshgrid(height, width)

    @staticmethod
    def _create_meshgrid(height: int, width: int) -> Tensor:
        grid: Tensor = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
        return convert_points_to_homogeneous(grid)  # append ones to last dim

    def compute_projection_matrix(self, pinhole_src: PinholeCamera) -> DepthWarper:
        r"""Compute the projection matrix from the source to destination frame."""
        if not isinstance(self._pinhole_dst, PinholeCamera):
            raise TypeError(
                f"Member self._pinhole_dst expected to be of class PinholeCamera. Got {type(self._pinhole_dst)}"
            )
        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError(f"Argument pinhole_src expected to be of class PinholeCamera. Got {type(pinhole_src)}")
        # compute the relative pose between the non reference and the reference
        # camera frames.
        dst_trans_src: Tensor = compose_transformations(
            self._pinhole_dst.extrinsics, inverse_transformation(pinhole_src.extrinsics)
        )

        # compute the projection matrix between the non reference cameras and
        # the reference.
        dst_proj_src: Tensor = torch.matmul(self._pinhole_dst.intrinsics, dst_trans_src)

        # update class members
        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src
        return self

    def _compute_projection(self, x: float, y: float, invd: float) -> Tensor:
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        point = tensor([[[x], [y], [invd], [1.0]]], device=self._dst_proj_src.device, dtype=self._dst_proj_src.dtype)
        flow = torch.matmul(self._dst_proj_src, point)
        z = 1.0 / flow[:, 2]
        _x = flow[:, 0] * z
        _y = flow[:, 1] * z
        return kornia_ops.concatenate([_x, _y], 1)

    def compute_subpixel_step(self) -> Tensor:
        """Compute the required inverse depth step to achieve sub pixel accurate sampling of the depth cost volume,
        per camera.

        Szeliski, Richard, and Daniel Scharstein. "Symmetric sub-pixel stereo matching." European Conference on Computer
        Vision. Springer Berlin Heidelberg, 2002.
        """
        delta_d = 0.01
        xy_m1 = self._compute_projection(self.width / 2, self.height / 2, 1.0 - delta_d)
        xy_p1 = self._compute_projection(self.width / 2, self.height / 2, 1.0 + delta_d)
        dx = torch.norm((xy_p1 - xy_m1), 2, dim=-1) / 2.0
        dxdd = dx / (delta_d)  # pixel*(1/meter)
        # half pixel sampling, we're interested in the min for all cameras
        return torch.min(0.5 / dxdd)

    def warp_grid(self, depth_src: Tensor) -> Tensor:
        """Compute a grid for warping a given the depth from the reference pinhole camera.

        The function `compute_projection_matrix` has to be called beforehand in order to have precomputed the relative
        projection matrices encoding the relative pose and the intrinsics between the reference and a non reference
        camera.
        """
        # TODO: add type and value checkings
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        if len(depth_src.shape) != 4:
            raise ValueError(f"Input depth_src has to be in the shape of Bx1xHxW. Got {depth_src.shape}")

        # unpack depth attributes
        batch_size, _, _, _ = depth_src.shape
        device: torch.device = depth_src.device
        dtype: torch.dtype = depth_src.dtype

        # expand the base coordinate grid according to the input batch size
        pixel_coords: Tensor = self.grid.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1)  # BxHxWx3

        # reproject the pixel coordinates to the camera frame
        cam_coords_src: Tensor = pixel2cam(
            depth_src, self._pinhole_src.intrinsics_inverse().to(device=device, dtype=dtype), pixel_coords
        )  # BxHxWx3

        # reproject the camera coordinates to the pixel
        pixel_coords_src: Tensor = cam2pixel(
            cam_coords_src, self._dst_proj_src.to(device=device, dtype=dtype)
        )  # (B*N)xHxWx2

        # normalize between -1 and 1 the coordinates
        pixel_coords_src_norm: Tensor = normalize_pixel_coordinates(pixel_coords_src, self.height, self.width)
        return pixel_coords_src_norm

    def forward(self, depth_src: Tensor, patch_dst: Tensor) -> Tensor:
        """Warp a tensor from destination frame to reference given the depth in the reference frame.

        Args:
            depth_src: the depth in the reference frame. The tensor must have a shape :math:`(B, 1, H, W)`.
            patch_dst: the patch in the destination frame. The tensor must have a shape :math:`(B, C, H, W)`.

        Return:
            the warped patch from destination frame to reference.

        Shape:
            - Output: :math:`(N, C, H, W)` where C = number of channels.

        Example:
            >>> # pinholes camera models
            >>> pinhole_dst = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
            ... torch.tensor([32]), torch.tensor([32]))
            >>> pinhole_src = PinholeCamera(torch.randn(1, 4, 4), torch.randn(1, 4, 4),
            ... torch.tensor([32]), torch.tensor([32]))
            >>> # create the depth warper, compute the projection matrix
            >>> warper = DepthWarper(pinhole_dst, 32, 32)
            >>> _ = warper.compute_projection_matrix(pinhole_src)
            >>> # warp the destination frame to reference by depth
            >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
            >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
            >>> image_src = warper(depth_src, image_dst)  # NxCxHxW
        """
        return kornia_ops.map_coordinates(
            patch_dst,
            self.warp_grid(depth_src),
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

