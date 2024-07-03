class LongestMaxSize(DualTransform):
    """Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): maximum size of the image after the transformation. When using a list, max size
            will be randomly selected from the values in the list.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(MaxSizeInitSchema):
        pass

    def __init__(
        self,
        max_size: int | Sequence[int] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 1,
    ):
        super().__init__(p, always_apply)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.longest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        max_size: int,
        **params: Any,
    ) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]

        scale = max_size / max([height, width])
        return fgeometric.keypoint_scale(keypoint, scale, scale)

    def get_params(self) -> dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("max_size", "interpolation")class SmallestMaxSize(DualTransform):
    """Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): maximum size of smallest side of the image after the transformation. When using a
            list, max size will be randomly selected from the values in the list.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(MaxSizeInitSchema):
        pass

    def __init__(
        self,
        max_size: int | Sequence[int] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 1,
    ):
        super().__init__(p, always_apply)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.smallest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return bbox

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        max_size: int,
        **params: Any,
    ) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]

        scale = max_size / min([height, width])
        return fgeometric.keypoint_scale(keypoint, scale, scale)

    def get_params(self) -> dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("max_size", "interpolation")