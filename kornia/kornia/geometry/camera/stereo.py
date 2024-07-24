class StereoCamera:
    def __init__(self, rectified_left_camera: Tensor, rectified_right_camera: Tensor) -> None:
        r"""Class representing a horizontal stereo camera setup.

        Args:
            rectified_left_camera: The rectified left camera projection matrix
              of shape :math:`(B, 3, 4)`
            rectified_right_camera: The rectified right camera projection matrix
              of shape :math:`(B, 3, 4)`
        """
        self._check_stereo_camera(rectified_left_camera, rectified_right_camera)
        self.rectified_left_camera: Tensor = rectified_left_camera
        self.rectified_right_camera: Tensor = rectified_right_camera

        self.device = self.rectified_left_camera.device
        self.dtype = self.rectified_left_camera.dtype

        self._Q_matrix = self._init_Q_matrix()

    @staticmethod
    def _check_stereo_camera(rectified_left_camera: Tensor, rectified_right_camera: Tensor) -> None:
        r"""Utility function to ensure user specified correct camera matrices.

        Args:
            rectified_left_camera: The rectified left camera projection matrix
              of shape :math:`(B, 3, 4)`
            rectified_right_camera: The rectified right camera projection matrix
              of shape :math:`(B, 3, 4)`
        """
        # Ensure correct shapes
        if len(rectified_left_camera.shape) != 3:
            raise StereoException(
                f"Expected 'rectified_left_camera' to have 3 dimensions. Got {rectified_left_camera.shape}."
            )

        if len(rectified_right_camera.shape) != 3:
            raise StereoException(
                f"Expected 'rectified_right_camera' to have 3 dimension. Got {rectified_right_camera.shape}."
            )

        if rectified_left_camera.shape[:1] == (3, 4):
            raise StereoException(
                f"Expected each 'rectified_left_camera' to be of shape (3, 4).Got {rectified_left_camera.shape[:1]}."
            )

        if rectified_right_camera.shape[:1] == (3, 4):
            raise StereoException(
                f"Expected each 'rectified_right_camera' to be of shape (3, 4).Got {rectified_right_camera.shape[:1]}."
            )

        # Ensure same devices for cameras.
        if rectified_left_camera.device != rectified_right_camera.device:
            raise StereoException(
                "Expected 'rectified_left_camera' and 'rectified_right_camera' "
                "to be on the same devices."
                f"Got {rectified_left_camera.device} and {rectified_right_camera.device}."
            )

        # Ensure same dtypes for cameras.
        if rectified_left_camera.dtype != rectified_right_camera.dtype:
            raise StereoException(
                "Expected 'rectified_left_camera' and 'rectified_right_camera' to"
                "have same dtype."
                f"Got {rectified_left_camera.dtype} and {rectified_right_camera.dtype}."
            )

        # Ensure all intrinsics parameters (fx, fy, cx, cy) are the same in both cameras.
        if not torch.all(torch.eq(rectified_left_camera[..., :, :3], rectified_right_camera[..., :, :3])):
            raise StereoException(
                "Expected 'left_rectified_camera' and 'rectified_right_camera' to have"
                "same parameters except for the last column."
                f"Got {rectified_left_camera[..., :, :3]} and {rectified_right_camera[..., :, :3]}."
            )

        # Ensure that tx * fx is negative and exists.
        tx_fx = rectified_right_camera[..., 0, 3]
        if torch.all(torch.gt(tx_fx, 0)):
            raise StereoException(f"Expected :math:`T_x * f_x` to be negative. Got {tx_fx}.")

    @property
    def batch_size(self) -> int:
        r"""Return the batch size of the storage.

        Returns:
           scalar with the batch size
        """
        return self.rectified_left_camera.shape[0]

    @property
    def fx(self) -> Tensor:
        r"""Return the focal length in the x-direction.

        Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 0, 0]

    @property
    def fy(self) -> Tensor:
        r"""Returns the focal length in the y-direction.

        Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 1, 1]

    @property
    def cx_left(self) -> Tensor:
        r"""Return the x-coordinate of the principal point for the left camera.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 0, 2]

    @property
    def cx_right(self) -> Tensor:
        r"""Return the x-coordinate of the principal point for the right camera.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_right_camera[..., 0, 2]

    @property
    def cy(self) -> Tensor:
        r"""Return the y-coordinate of the principal point.

        Note that the y-coordinate of the principal points
        is assumed to be equal for the left and right camera.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 1, 2]

    @property
    def tx(self) -> Tensor:
        r"""The horizontal baseline between the two cameras.

        Returns:
            Tensor of shape :math:`(B)`
        """
        return -self.rectified_right_camera[..., 0, 3] / self.fx

    @property
    def Q(self) -> Tensor:
        r"""The Q matrix of the horizontal stereo setup.

        This matrix is used for reprojecting a disparity tensor to
        the corresponding point cloud. Note that this is in a general form that allows different focal
        lengths in the x and y direction.

        Return:
            The Q matrix of shape :math:`(B, 4, 4)`.
        """
        return self._Q_matrix

    def _init_Q_matrix(self) -> Tensor:
        r"""Initialized the Q matrix of the horizontal stereo setup. See the Q property.

        Returns:
            The Q matrix of shape :math:`(B, 4, 4)`.
        """
        Q = zeros((self.batch_size, 4, 4), device=self.device, dtype=self.dtype)
        baseline: Tensor = -self.tx
        Q[:, 0, 0] = self.fy * baseline
        Q[:, 0, 3] = -self.fy * self.cx_left * baseline
        Q[:, 1, 1] = self.fx * baseline
        Q[:, 1, 3] = -self.fx * self.cy * baseline
        Q[:, 2, 3] = self.fx * self.fy * baseline
        Q[:, 3, 2] = -self.fy
        Q[:, 3, 3] = self.fy * (self.cx_left - self.cx_right)  # NOTE: This is usually zero.
        return Q

    def reproject_disparity_to_3D(self, disparity_tensor: Tensor) -> Tensor:
        r"""Reproject the disparity tensor to a 3D point cloud.

        Args:
            disparity_tensor: Disparity tensor of shape :math:`(B, 1, H, W)`.

        Returns:
            The 3D point cloud of shape :math:`(B, H, W, 3)`
        """
        return reproject_disparity_to_3D(disparity_tensor, self.Q)

