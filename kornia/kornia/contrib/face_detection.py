class FaceDetectorResult:
    r"""Encapsulate the results obtained by the :py:class:`kornia.contrib.FaceDetector`.

    Args:
        data: the encoded results coming from the feature detector with shape :math:`(14,)`.
    """

    def __init__(self, data: torch.Tensor) -> None:
        if len(data) < 15:
            raise ValueError(f"Result must comes as vector of size(14). Got: {data.shape}.")
        self._data = data

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "FaceDetectorResult":
        """Like :func:`torch.nn.Module.to()` method."""
        self._data = self._data.to(device=device, dtype=dtype)
        return self

    @property
    def xmin(self) -> torch.Tensor:
        """The bounding box top-left x-coordinate."""
        return self._data[..., 0]

    @property
    def ymin(self) -> torch.Tensor:
        """The bounding box top-left y-coordinate."""
        return self._data[..., 1]

    @property
    def xmax(self) -> torch.Tensor:
        """The bounding box bottom-right x-coordinate."""
        return self._data[..., 2]

    @property
    def ymax(self) -> torch.Tensor:
        """The bounding box bottom-right y-coordinate."""
        return self._data[..., 3]

    def get_keypoint(self, keypoint: FaceKeypoint) -> torch.Tensor:
        """The [x y] position of a given facial keypoint.

        Args:
            keypoint: the keypoint type to return the position.
        """
        if keypoint == FaceKeypoint.EYE_LEFT:
            out = self._data[..., (4, 5)]
        elif keypoint == FaceKeypoint.EYE_RIGHT:
            out = self._data[..., (6, 7)]
        elif keypoint == FaceKeypoint.NOSE:
            out = self._data[..., (8, 9)]
        elif keypoint == FaceKeypoint.MOUTH_LEFT:
            out = self._data[..., (10, 11)]
        elif keypoint == FaceKeypoint.MOUTH_RIGHT:
            out = self._data[..., (12, 13)]
        else:
            raise ValueError(f"Not valid keypoint type. Got: {keypoint}.")
        return out

    @property
    def score(self) -> torch.Tensor:
        """The detection score."""
        return self._data[..., 14]

    @property
    def width(self) -> torch.Tensor:
        """The bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self) -> torch.Tensor:
        """The bounding box height."""
        return self.ymax - self.ymin

    @property
    def top_left(self) -> torch.Tensor:
        """The [x y] position of the top-left coordinate of the bounding box."""
        return self._data[..., (0, 1)]

    @property
    def top_right(self) -> torch.Tensor:
        """The [x y] position of the top-left coordinate of the bounding box."""
        out = self.top_left
        out[..., 0] += self.width
        return out

    @property
    def bottom_right(self) -> torch.Tensor:
        """The [x y] position of the bottom-right coordinate of the bounding box."""
        return self._data[..., (2, 3)]

    @property
    def bottom_left(self) -> torch.Tensor:
        """The [x y] position of the top-left coordinate of the bounding box."""
        out = self.top_left
        out[..., 1] += self.height
        return out