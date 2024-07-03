class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit: range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        crop_border (bool): If True would make a largest possible crop within rotated image
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(RotateInitSchema):
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box"
        crop_border: bool = Field(
            default=False,
            description="If True, makes a largest possible crop within the rotated image.",
        )

    def __init__(
        self,
        limit: ScaleFloatType = (-90, 90),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        crop_border: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.limit = cast(Tuple[float, float], limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.crop_border = crop_border

    def apply(
        self,
        img: np.ndarray,
        angle: float,
        interpolation: int,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        img_out = fgeometric.rotate(img, angle, interpolation, self.border_mode, self.value)
        if self.crop_border:
            return fcrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_mask(
        self,
        mask: np.ndarray,
        angle: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        img_out = fgeometric.rotate(mask, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)
        if self.crop_border:
            return fcrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        angle: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        cols: int,
        rows: int,
        **params: Any,
    ) -> np.ndarray:
        bbox_out = fgeometric.bbox_rotate(bbox, angle, self.rotate_method, rows, cols)
        if self.crop_border:
            return fcrops.crop_bbox_by_coords(bbox_out, (x_min, y_min, x_max, y_max), rows, cols)
        return bbox_out

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        angle: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        cols: int,
        rows: int,
        **params: Any,
    ) -> KeypointInternalType:
        keypoint_out = fgeometric.keypoint_rotate(keypoint, angle, rows, cols, **params)
        if self.crop_border:
            return fcrops.crop_keypoint_by_coords(keypoint_out, (x_min, y_min, x_max, y_max))
        return keypoint_out

    @staticmethod
    def _rotated_rect_with_max_area(height: int, width: int, angle: float) -> dict[str, int]:
        """Given a rectangle of size wxh that has been rotated by 'angle' (in
        degrees), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.

        Reference:
            https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        angle = math.radians(angle)
        width_is_longer = width >= height
        side_long, side_short = (width, height) if width_is_longer else (height, width)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # it is sufficient to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < SMALL_NUMBER:
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a

        return {
            "x_min": max(0, int(width / 2 - wr / 2)),
            "x_max": min(width, int(width / 2 + wr / 2)),
            "y_min": max(0, int(height / 2 - hr / 2)),
            "y_max": min(height, int(height / 2 + hr / 2)),
        }

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        out_params = {"angle": random.uniform(self.limit[0], self.limit[1])}
        if self.crop_border:
            height, width = params["image"].shape[:2]
            out_params.update(self._rotated_rect_with_max_area(height, width, out_params["angle"]))
        else:
            out_params.update({"x_min": -1, "x_max": -1, "y_min": -1, "y_max": -1})

        return out_params

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("limit", "interpolation", "border_mode", "value", "mask_value", "rotate_method", "crop_border")