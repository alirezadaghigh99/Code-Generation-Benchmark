class PinholeCamera:
    r"""Class that represents a Pinhole Camera model.

    Args:
        intrinsics: tensor with shape :math:`(B, 4, 4)`
          containing the full 4x4 camera calibration matrix.
        extrinsics: tensor with shape :math:`(B, 4, 4)`
          containing the full 4x4 rotation-translation matrix.
        height: tensor with shape :math:`(B)` containing the image height.
        width: tensor with shape :math:`(B)` containing the image width.

    .. note::
        We assume that the class attributes are in batch form in order to take
        advantage of PyTorch parallelism to boost computing performance.
    """

    def __init__(self, intrinsics: Tensor, extrinsics: Tensor, height: Tensor, width: Tensor) -> None:
        # verify batch size and shapes
        self._check_valid([intrinsics, extrinsics, height, width])
        self._check_valid_params(intrinsics, "intrinsics")
        self._check_valid_params(extrinsics, "extrinsics")
        self._check_valid_shape(height, "height")
        self._check_valid_shape(width, "width")
        self._check_consistent_device([intrinsics, extrinsics, height, width])
        # set class attributes
        self.height: Tensor = height
        self.width: Tensor = width
        self._intrinsics: Tensor = intrinsics
        self._extrinsics: Tensor = extrinsics

    @staticmethod
    def _check_valid(data_iter: Iterable[Tensor]) -> bool:
        if not all(data.shape[0] for data in data_iter):
            raise ValueError("Arguments shapes must match")
        return True

    @staticmethod
    def _check_valid_params(data: Tensor, data_name: str) -> bool:
        if len(data.shape) not in (3, 4) and data.shape[-2:] != (4, 4):  # Shouldn't this be an OR logic than AND?
            raise ValueError(
                f"Argument {data_name} shape must be in the following shape Bx4x4 or BxNx4x4. Got {data.shape}"
            )
        return True

    @staticmethod
    def _check_valid_shape(data: Tensor, data_name: str) -> bool:
        if not len(data.shape) == 1:
            raise ValueError(f"Argument {data_name} shape must be in the following shape B. Got {data.shape}")
        return True

    @staticmethod
    def _check_consistent_device(data_iter: List[Tensor]) -> None:
        first = data_iter[0]
        for data in data_iter:
            KORNIA_CHECK_SAME_DEVICE(data, first)

    def device(self) -> torch.device:
        r"""Returns the device for camera buffers.

        Returns:
            Device type
        """
        return self._intrinsics.device

    @property
    def intrinsics(self) -> Tensor:
        r"""The full 4x4 intrinsics matrix.

        Returns:
            tensor of shape :math:`(B, 4, 4)`.
        """
        if not self._check_valid_params(self._intrinsics, "intrinsics"):
            raise AssertionError
        return self._intrinsics

    @property
    def extrinsics(self) -> Tensor:
        r"""The full 4x4 extrinsics matrix.

        Returns:
            tensor of shape :math:`(B, 4, 4)`.
        """
        if not self._check_valid_params(self._extrinsics, "extrinsics"):
            raise AssertionError
        return self._extrinsics

    @property
    def batch_size(self) -> int:
        r"""Return the batch size of the storage.

        Returns:
            scalar with the batch size.
        """
        return self.intrinsics.shape[0]

    @property
    def fx(self) -> Tensor:
        r"""Return the focal length in the x-direction.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.intrinsics[..., 0, 0]

    @property
    def fy(self) -> Tensor:
        r"""Return the focal length in the y-direction.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.intrinsics[..., 1, 1]

    @property
    def cx(self) -> Tensor:
        r"""Return the x-coordinate of the principal point.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.intrinsics[..., 0, 2]

    @property
    def cy(self) -> Tensor:
        r"""Return the y-coordinate of the principal point.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.intrinsics[..., 1, 2]

    @property
    def tx(self) -> Tensor:
        r"""Return the x-coordinate of the translation vector.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.extrinsics[..., 0, -1]

    @tx.setter
    def tx(self, value: Union[Tensor, float]) -> "PinholeCamera":
        r"""Set the x-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 0, -1] = value
        return self

    @property
    def ty(self) -> Tensor:
        r"""Return the y-coordinate of the translation vector.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.extrinsics[..., 1, -1]

    @ty.setter
    def ty(self, value: Union[Tensor, float]) -> "PinholeCamera":
        r"""Set the y-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 1, -1] = value
        return self

    @property
    def tz(self) -> Tensor:
        r"""Returns the z-coordinate of the translation vector.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.extrinsics[..., 2, -1]

    @tz.setter
    def tz(self, value: Union[Tensor, float]) -> "PinholeCamera":
        r"""Set the y-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 2, -1] = value
        return self

    @property
    def rt_matrix(self) -> Tensor:
        r"""Return the 3x4 rotation-translation matrix.

        Returns:
            tensor of shape :math:`(B, 3, 4)`.
        """
        return self.extrinsics[..., :3, :4]

    @property
    def camera_matrix(self) -> Tensor:
        r"""Return the 3x3 camera matrix containing the intrinsics.

        Returns:
            tensor of shape :math:`(B, 3, 3)`.
        """
        return self.intrinsics[..., :3, :3]

    @property
    def rotation_matrix(self) -> Tensor:
        r"""Return the 3x3 rotation matrix from the extrinsics.

        Returns:
            tensor of shape :math:`(B, 3, 3)`.
        """
        return self.extrinsics[..., :3, :3]

    @property
    def translation_vector(self) -> Tensor:
        r"""Return the translation vector from the extrinsics.

        Returns:
            tensor of shape :math:`(B, 3, 1)`.
        """
        return self.extrinsics[..., :3, -1:]

    def clone(self) -> "PinholeCamera":
        r"""Return a deep copy of the current object instance."""
        height: Tensor = self.height.clone()
        width: Tensor = self.width.clone()
        intrinsics: Tensor = self.intrinsics.clone()
        extrinsics: Tensor = self.extrinsics.clone()
        return PinholeCamera(intrinsics, extrinsics, height, width)

    def intrinsics_inverse(self) -> Tensor:
        r"""Return the inverse of the 4x4 instrisics matrix.

        Returns:
            tensor of shape :math:`(B, 4, 4)`.
        """
        return self.intrinsics.inverse()

    def scale(self, scale_factor: Tensor) -> "PinholeCamera":
        r"""Scale the pinhole model.

        Args:
            scale_factor: a tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              :math:`(B)` or :math:`(1)`.

        Returns:
            the camera model with scaled parameters.
        """
        # scale the intrinsic parameters
        intrinsics: Tensor = self.intrinsics.clone()
        intrinsics[..., 0, 0] *= scale_factor
        intrinsics[..., 1, 1] *= scale_factor
        intrinsics[..., 0, 2] *= scale_factor
        intrinsics[..., 1, 2] *= scale_factor
        # scale the image height/width
        height: Tensor = scale_factor * self.height.clone()
        width: Tensor = scale_factor * self.width.clone()
        return PinholeCamera(intrinsics, self.extrinsics, height, width)

    def scale_(self, scale_factor: Union[float, Tensor]) -> "PinholeCamera":
        r"""Scale the pinhole model in-place.

        Args:
            scale_factor: a tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              :math:`(B)` or :math:`(1)`.

        Returns:
            the camera model with scaled parameters.
        """
        # scale the intrinsic parameters
        self.intrinsics[..., 0, 0] *= scale_factor
        self.intrinsics[..., 1, 1] *= scale_factor
        self.intrinsics[..., 0, 2] *= scale_factor
        self.intrinsics[..., 1, 2] *= scale_factor
        # scale the image height/width
        self.height *= scale_factor
        self.width *= scale_factor
        return self

    def project(self, point_3d: Tensor) -> Tensor:
        r"""Project a 3d point in world coordinates onto the 2d camera plane.

        Args:
            point3d: tensor containing the 3d points to be projected
                to the camera plane. The shape of the tensor can be :math:`(*, 3)`.

        Returns:
            tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

        Example:
            >>> _ = torch.manual_seed(0)
            >>> X = torch.rand(1, 3)
            >>> K = torch.eye(4)[None]
            >>> E = torch.eye(4)[None]
            >>> h = torch.ones(1)
            >>> w = torch.ones(1)
            >>> pinhole = kornia.geometry.camera.PinholeCamera(K, E, h, w)
            >>> pinhole.project(X)
            tensor([[5.6088, 8.6827]])
        """
        P = self.intrinsics @ self.extrinsics
        return convert_points_from_homogeneous(transform_points(P, point_3d))

    def unproject(self, point_2d: Tensor, depth: Tensor) -> Tensor:
        r"""Unproject a 2d point in 3d.

        Transform coordinates in the pixel frame to the world frame.

        Args:
            point2d: tensor containing the 2d to be projected to
                world coordinates. The shape of the tensor can be :math:`(*, 2)`.
            depth: tensor containing the depth value of each 2d
                points. The tensor shape must be equal to point2d :math:`(*, 1)`.
            normalize: whether to normalize the pointcloud. This
                must be set to `True` when the depth is represented as the Euclidean
                ray length from the camera position.

        Returns:
            tensor of (x, y, z) world coordinates with shape :math:`(*, 3)`.

        Example:
            >>> _ = torch.manual_seed(0)
            >>> x = torch.rand(1, 2)
            >>> depth = torch.ones(1, 1)
            >>> K = torch.eye(4)[None]
            >>> E = torch.eye(4)[None]
            >>> h = torch.ones(1)
            >>> w = torch.ones(1)
            >>> pinhole = kornia.geometry.camera.PinholeCamera(K, E, h, w)
            >>> pinhole.unproject(x, depth)
            tensor([[0.4963, 0.7682, 1.0000]])
        """
        P = self.intrinsics @ self.extrinsics
        P_inv = _torch_inverse_cast(P)
        return transform_points(P_inv, convert_points_to_homogeneous(point_2d) * depth)

    # NOTE: just for test. Decide if we keep it.
    @classmethod
    def from_parameters(
        self,
        fx: Tensor,
        fy: Tensor,
        cx: Tensor,
        cy: Tensor,
        height: int,
        width: int,
        tx: Tensor,
        ty: Tensor,
        tz: Tensor,
        batch_size: int,
        device: Device,
        dtype: torch.dtype,
    ) -> "PinholeCamera":
        # create the camera matrix
        intrinsics = zeros(batch_size, 4, 4, device=device, dtype=dtype)
        intrinsics[..., 0, 0] += fx
        intrinsics[..., 1, 1] += fy
        intrinsics[..., 0, 2] += cx
        intrinsics[..., 1, 2] += cy
        intrinsics[..., 2, 2] += 1.0
        intrinsics[..., 3, 3] += 1.0
        # create the pose matrix
        extrinsics = eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        extrinsics[..., 0, -1] += tx
        extrinsics[..., 1, -1] += ty
        extrinsics[..., 2, -1] += tz
        # create image hegith and width
        height_tmp = zeros(batch_size, device=device, dtype=dtype)
        height_tmp[..., 0] += height
        width_tmp = zeros(batch_size, device=device, dtype=dtype)
        width_tmp[..., 0] += width
        return self(intrinsics, extrinsics, height_tmp, width_tmp)