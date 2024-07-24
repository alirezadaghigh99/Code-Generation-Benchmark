class RandomResizedCrop(_BaseRandomSizedCrop):
    """Torchvision's variant of crop a random part of the input and rescale it to some size.

    Args:
        size (int, int): expected output size of the crop, for each edge. If size is an int instead of sequence
            like (height, width), a square output size (size, size) is made. If provided a sequence of length 1,
            it will be interpreted as (size[0], size[0]).
        scale ((float, float)): Specifies the lower and upper bounds for the random area of the crop, before resizing.
            The scale is defined with respect to the area of the original image.
        ratio ((float, float)): lower and upper bounds for the random aspect ratio of the crop, before resizing.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        scale: Annotated[tuple[float, float], AfterValidator(check_01)] = (0.08, 1.0)
        ratio: Annotated[tuple[float, float], AfterValidator(check_0plus)] = (0.75, 1.3333333333333333)
        width: int | None = Field(
            None,
            deprecated="Initializing with 'height' and 'width' is deprecated. Use size instead.",
        )
        height: int | None = Field(
            None,
            deprecated="Initializing with 'height' and 'width' is deprecated. Use size instead.",
        )
        size: ScaleIntType | None = None
        p: ProbabilityType = 1
        interpolation: InterpolationType = cv2.INTER_LINEAR

        @model_validator(mode="after")
        def process(self) -> Self:
            if isinstance(self.size, int):
                if isinstance(self.width, int):
                    self.size = (self.size, self.width)
                else:
                    msg = "If size is an integer, width as integer must be specified."
                    raise TypeError(msg)

            if self.size is None:
                if self.height is None or self.width is None:
                    message = "If 'size' is not provided, both 'height' and 'width' must be specified."
                    raise ValueError(message)
                self.size = (self.height, self.width)

            return self

    def __init__(
        self,
        # NOTE @zetyquickly: when (width, height) are deprecated, make 'size' non optional
        size: ScaleIntType | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(size=cast(Tuple[int, int], size), interpolation=interpolation, p=p, always_apply=always_apply)
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, tuple[int, int, int, int]]:
        img = params["image"]
        image_height, image_width = img.shape[:2]
        area = image_height * image_width

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            width = int(round(math.sqrt(target_area * aspect_ratio)))
            height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < width <= image_width and 0 < height <= image_height:
                i = random.randint(0, image_height - height)
                j = random.randint(0, image_width - width)

                h_start = i * 1.0 / (image_height - height + 1e-10)
                w_start = j * 1.0 / (image_width - width + 1e-10)

                crop_coords = fcrops.get_crop_coords(image_height, image_width, height, width, h_start, w_start)

                return {"crop_coords": crop_coords}

        # Fallback to central crop
        in_ratio = image_width / image_height
        if in_ratio < min(self.ratio):
            width = image_width
            height = int(round(image_width / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            height = image_height
            width = int(round(height * max(self.ratio)))
        else:  # whole image
            width = image_width
            height = image_height

        i = (image_height - height) // 2
        j = (image_width - width) // 2

        h_start = i * 1.0 / (image_height - height + 1e-10)
        w_start = j * 1.0 / (image_width - width + 1e-10)

        crop_coords = fcrops.get_crop_coords(image_height, image_width, height, width, h_start, w_start)

        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "size", "scale", "ratio", "interpolation"

class Crop(_BaseCrop):
    """Crop region from image.

    Args:
        x_min: Minimum upper left x coordinate.
        y_min: Minimum upper left y coordinate.
        x_max: Maximum lower right x coordinate.
        y_max: Maximum lower right y coordinate.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    class InitSchema(BaseTransformInitSchema):
        x_min: Annotated[int, Field(ge=0, description="Minimum upper left x coordinate")]
        y_min: Annotated[int, Field(ge=0, description="Minimum upper left y coordinate")]
        x_max: Annotated[int, Field(gt=0, description="Maximum lower right x coordinate")]
        y_max: Annotated[int, Field(gt=0, description="Maximum lower right y coordinate")]
        p: ProbabilityType = 1

        @model_validator(mode="after")
        def validate_coordinates(self) -> Self:
            if not self.x_min < self.x_max:
                msg = "x_max must be greater than x_min"
                raise ValueError(msg)
            if not self.y_min < self.y_max:
                msg = "y_max must be greater than y_min"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        x_min: int = 0,
        y_min: int = 0,
        x_max: int = 1024,
        y_max: int = 1024,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "x_min", "y_min", "x_max", "y_max"

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, tuple[int, int, int, int]]:
        return {"crop_coords": (self.x_min, self.y_min, self.x_max, self.y_max)}

class RandomSizedCrop(_BaseRandomSizedCrop):
    """Crop a random portion of the input and rescale it to a specific size.

    Args:
        min_max_height ((int, int)): crop size limits.
        size ((int, int)): target size for the output image, i.e. (height, width) after crop and resize
        w2h_ratio (float): aspect ratio of crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        interpolation: InterpolationType = cv2.INTER_LINEAR
        p: ProbabilityType = 1
        min_max_height: OnePlusIntRangeType
        w2h_ratio: Annotated[float, Field(gt=0, description="Aspect ratio of crop.")]
        width: int | None = Field(
            None,
            deprecated=(
                "Initializing with 'size' as an integer and a separate 'width' is deprecated. "
                "Please use a tuple (height, width) for the 'size' argument."
            ),
        )
        height: int | None = Field(
            None,
            deprecated=(
                "Initializing with 'height' and 'width' is deprecated. "
                "Please use a tuple (height, width) for the 'size' argument."
            ),
        )
        size: ScaleIntType | None = None

        @model_validator(mode="after")
        def process(self) -> Self:
            if isinstance(self.size, int):
                if isinstance(self.width, int):
                    self.size = (self.size, self.width)
                else:
                    msg = "If size is an integer, width as integer must be specified."
                    raise TypeError(msg)

            if self.size is None:
                if self.height is None or self.width is None:
                    message = "If 'size' is not provided, both 'height' and 'width' must be specified."
                    raise ValueError(message)
                self.size = (self.height, self.width)
            return self

    def __init__(
        self,
        min_max_height: tuple[int, int],
        # NOTE @zetyquickly: when (width, height) are deprecated, make 'size' non optional
        size: ScaleIntType | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        w2h_ratio: float = 1.0,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(size=cast(Tuple[int, int], size), interpolation=interpolation, p=p, always_apply=always_apply)
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, tuple[int, int, int, int]]:
        image_height, image_width = params["image"].shape[:2]

        crop_height = random.randint(self.min_max_height[0], self.min_max_height[1])
        crop_width = int(crop_height * self.w2h_ratio)

        h_start = random.random()
        w_start = random.random()

        crop_coords = fcrops.get_crop_coords(image_height, image_width, crop_height, crop_width, h_start, w_start)

        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "min_max_height", "size", "w2h_ratio", "interpolation"

class CropAndPad(DualTransform):
    """Crop and pad images by pixel amounts or fractions of image sizes.
    Cropping removes pixels at the sides (i.e., extracts a subimage from a given full image).
    Padding adds pixels to the sides (e.g., black pixels).
    This transformation will never crop images below a height or width of 1.

    Note:
        This transformation automatically resizes images back to their original size. To deactivate this, add the
        parameter `keep_size=False`.

    Args:
        px (int,
            tuple[int, int],
            tuple[int, int, int, int],
            tuple[Union[int, tuple[int, int], list[int]],
                  Union[int, tuple[int, int], list[int]],
                  Union[int, tuple[int, int], list[int]],
                  Union[int, tuple[int, int], list[int]]]):
            The number of pixels to crop (negative values) or pad (positive values) on each side of the image.
                Either this or the parameter `percent` may be set, not both at the same time.

                * If `None`, then pixel-based cropping/padding will not be used.
                * If `int`, then that exact number of pixels will always be cropped/padded.
                * If a `tuple` of two `int`s with values `a` and `b`, then each side will be cropped/padded by a
                    random amount sampled uniformly per image and side from the interval `[a, b]`.
                    If `sample_independently` is set to `False`, only one value will be sampled per
                        image and used for all sides.
                * If a `tuple` of four entries, then the entries represent top, right, bottom, and left.
                    Each entry may be:
                    - A single `int` (always crop/pad by exactly that value).
                    - A `tuple` of two `int`s `a` and `b` (crop/pad by an amount within `[a, b]`).
                    - A `list` of `int`s (crop/pad by a random value that is contained in the `list`).

        percent (float,
                 tuple[float, float],
                 tuple[float, float, float, float],
                 tuple[Union[float, tuple[float, float], list[float]],
                       Union[float, tuple[float, float], list[float]],
                       Union[float, tuple[float, float], list[float]],
                       Union[float, tuple[float, float], list[float]]]):
            The number of pixels to crop (negative values) or pad (positive values) on each side of the image given
                as a *fraction* of the image height/width. E.g. if this is set to `-0.1`, the transformation will
                always crop away `10%` of the image's height at both the top and the bottom (both `10%` each),
                as well as `10%` of the width at the right and left. Expected value range is `(-1.0, inf)`.
                Either this or the parameter `px` may be set, not both at the same time.

                * If `None`, then fraction-based cropping/padding will not be used.
                * If `float`, then that fraction will always be cropped/padded.
                * If a `tuple` of two `float`s with values `a` and `b`, then each side will be cropped/padded by a
                random fraction sampled uniformly per image and side from the interval `[a, b]`.
                If `sample_independently` is set to `False`, only one value will be sampled per image and used
                for all sides.
                * If a `tuple` of four entries, then the entries represent top, right, bottom, and left.
                    Each entry may be:
                    - A single `float` (always crop/pad by exactly that percent value).
                    - A `tuple` of two `float`s `a` and `b` (crop/pad by a fraction from `[a, b]`).
                    - A `list` of `float`s (crop/pad by a random value that is contained in the `list`).

        pad_mode (int): OpenCV border mode.
        pad_cval (Union[int, float, tuple[Union[int, float], Union[int, float]], list[Union[int, float]]]):
            The constant value to use if the pad mode is `BORDER_CONSTANT`.
                * If `number`, then that value will be used.
                * If a `tuple` of two numbers and at least one of them is a `float`, then a random number
                    will be uniformly sampled per image from the continuous interval `[a, b]` and used as the value.
                    If both numbers are `int`s, the interval is discrete.
                * If a `list` of numbers, then a random value will be chosen from the elements of the `list` and
                    used as the value.

        pad_cval_mask (Union[int, float, tuple[Union[int, float], Union[int, float]], list[Union[int, float]]]):
            Same as `pad_cval` but only for masks.

        keep_size (bool):
            After cropping and padding, the resulting image will usually have a different height/width compared to
            the original input image. If this parameter is set to `True`, then the cropped/padded image will be
            resized to the input image's size, i.e., the output shape is always identical to the input shape.

        sample_independently (bool):
            If `False` and the values for `px`/`percent` result in exactly one probability distribution for all
            image sides, only one single value will be sampled from that probability distribution and used for
            all sides. I.e., the crop/pad amount then is the same for all sides. If `True`, four values
            will be sampled independently, one per side.

        interpolation (int):
            OpenCV flag that is used to specify the interpolation algorithm for images. Should be one of:
            `cv2.INTER_NEAREST`, `cv2.INTER_LINEAR`, `cv2.INTER_CUBIC`, `cv2.INTER_AREA`, `cv2.INTER_LANCZOS4`.
            Default: `cv2.INTER_LINEAR`.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        unit8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        px: PxType | None = Field(
            default=None,
            description="Number of pixels to crop (negative) or pad (positive).",
        )
        percent: PercentType | None = Field(
            default=None,
            description="Fraction of image size to crop (negative) or pad (positive).",
        )
        pad_mode: BorderModeType = cv2.BORDER_CONSTANT
        pad_cval: ColorType = Field(
            default=0,
            description="Padding value if pad_mode is BORDER_CONSTANT.",
        )
        pad_cval_mask: ColorType = Field(
            default=0,
            description="Padding value for masks if pad_mode is BORDER_CONSTANT.",
        )
        keep_size: bool = Field(
            default=True,
            description="Whether to resize the image back to the original size after cropping and padding.",
        )
        sample_independently: bool = Field(
            default=True,
            description="Whether to sample the crop/pad size independently for each side.",
        )
        interpolation: InterpolationType = cv2.INTER_LINEAR
        p: ProbabilityType = 1

        @model_validator(mode="after")
        def check_px_percent(self) -> Self:
            if self.px is None and self.percent is None:
                msg = "Both px and percent parameters cannot be None simultaneously."
                raise ValueError(msg)
            if self.px is not None and self.percent is not None:
                msg = "Only px or percent may be set!"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        px: int | list[int] | None = None,
        percent: float | list[float] | None = None,
        pad_mode: int = cv2.BORDER_CONSTANT,
        pad_cval: ColorType = 0,
        pad_cval_mask: ColorType = 0,
        keep_size: bool = True,
        sample_independently: bool = True,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.px = px
        self.percent = percent

        self.pad_mode = pad_mode
        self.pad_cval = pad_cval
        self.pad_cval_mask = pad_cval_mask

        self.keep_size = keep_size
        self.sample_independently = sample_independently

        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        pad_value: float,
        rows: int,
        cols: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_and_pad(
            img,
            crop_params,
            pad_params,
            pad_value,
            rows,
            cols,
            interpolation,
            self.pad_mode,
            self.keep_size,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        pad_value_mask: float,
        rows: int,
        cols: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_and_pad(
            mask,
            crop_params,
            pad_params,
            pad_value_mask,
            rows,
            cols,
            interpolation,
            self.pad_mode,
            self.keep_size,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        rows: int,
        cols: int,
        result_rows: int,
        result_cols: int,
        **params: Any,
    ) -> BoxInternalType:
        return fcrops.crop_and_pad_bbox(bbox, crop_params, pad_params, rows, cols, result_rows, result_cols)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        rows: int,
        cols: int,
        result_rows: int,
        result_cols: int,
        **params: Any,
    ) -> KeypointInternalType:
        return fcrops.crop_and_pad_keypoint(
            keypoint,
            crop_params,
            pad_params,
            rows,
            cols,
            result_rows,
            result_cols,
            self.keep_size,
        )

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    @staticmethod
    def __prevent_zero(val1: int, val2: int, max_val: int) -> tuple[int, int]:
        regain = abs(max_val) + 1
        regain1 = regain // 2
        regain2 = regain // 2
        if regain1 + regain2 < regain:
            regain1 += 1

        if regain1 > val1:
            diff = regain1 - val1
            regain1 = val1
            regain2 += diff
        elif regain2 > val2:
            diff = regain2 - val2
            regain2 = val2
            regain1 += diff

        return val1 - regain1, val2 - regain2

    @staticmethod
    def _prevent_zero(crop_params: list[int], height: int, width: int) -> list[int]:
        top, right, bottom, left = crop_params

        remaining_height = height - (top + bottom)
        remaining_width = width - (left + right)

        if remaining_height < 1:
            top, bottom = CropAndPad.__prevent_zero(top, bottom, height)
        if remaining_width < 1:
            left, right = CropAndPad.__prevent_zero(left, right, width)

        return [max(top, 0), max(right, 0), max(bottom, 0), max(left, 0)]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        height, width = params["image"].shape[:2]

        if self.px is not None:
            new_params = self._get_px_params()
        else:
            percent_params = self._get_percent_params()
            new_params = [
                int(percent_params[0] * height),
                int(percent_params[1] * width),
                int(percent_params[2] * height),
                int(percent_params[3] * width),
            ]

        pad_params = [max(i, 0) for i in new_params]

        crop_params = self._prevent_zero([-min(i, 0) for i in new_params], height, width)

        top, right, bottom, left = crop_params
        crop_params = [left, top, width - right, height - bottom]
        result_rows = crop_params[3] - crop_params[1]
        result_cols = crop_params[2] - crop_params[0]
        if result_cols == width and result_rows == height:
            crop_params = []

        top, right, bottom, left = pad_params
        pad_params = [top, bottom, left, right]
        if any(pad_params):
            result_rows += top + bottom
            result_cols += left + right
        else:
            pad_params = []

        return {
            "crop_params": crop_params or None,
            "pad_params": pad_params or None,
            "pad_value": None if pad_params is None else self._get_pad_value(self.pad_cval),
            "pad_value_mask": None if pad_params is None else self._get_pad_value(self.pad_cval_mask),
            "result_rows": result_rows,
            "result_cols": result_cols,
        }

    def _get_px_params(self) -> list[int]:
        if self.px is None:
            msg = "px is not set"
            raise ValueError(msg)

        if isinstance(self.px, int):
            params = [self.px] * 4
        elif len(self.px) == PAIR:
            if self.sample_independently:
                params = [random.randrange(*self.px) for _ in range(4)]
            else:
                px = random.randrange(*self.px)
                params = [px] * 4
        elif isinstance(self.px[0], int):
            params = self.px
        elif len(self.px[0]) == PAIR:
            params = [random.randrange(*i) for i in self.px]
        else:
            params = [random.choice(i) for i in self.px]

        return params

    def _get_percent_params(self) -> list[float]:
        if self.percent is None:
            msg = "percent is not set"
            raise ValueError(msg)

        if isinstance(self.percent, float):
            params = [self.percent] * 4
        elif len(self.percent) == PAIR:
            if self.sample_independently:
                params = [random.uniform(*self.percent) for _ in range(4)]
            else:
                px = random.uniform(*self.percent)
                params = [px] * 4
        elif isinstance(self.percent[0], (int, float)):
            params = self.percent
        elif len(self.percent[0]) == PAIR:
            params = [random.uniform(*i) for i in self.percent]
        else:
            params = [random.choice(i) for i in self.percent]

        return params  # params = [top, right, bottom, left]

    @staticmethod
    def _get_pad_value(
        pad_value: ColorType,
    ) -> ScalarType:
        if isinstance(pad_value, (int, float)):
            return pad_value

        if len(pad_value) == PAIR:
            a, b = pad_value
            if isinstance(a, int) and isinstance(b, int):
                return random.randint(a, b)

            return random.uniform(a, b)

        return random.choice(pad_value)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "px",
            "percent",
            "pad_mode",
            "pad_cval",
            "pad_cval_mask",
            "keep_size",
            "sample_independently",
            "interpolation",
        )

class CenterCrop(_BaseCrop):
    """Crop the central part of the input.

    Args:
        height: height of the crop.
        width: width of the crop.
        p: probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    class InitSchema(CropInitSchema):
        pass

    def __init__(self, height: int, width: int, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p, always_apply)
        self.height = height
        self.width = width

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "height", "width"

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, tuple[int, int, int, int]]:
        img = params["image"]

        image_height, image_width = img.shape[:2]
        crop_coords = fcrops.get_center_crop_coords(image_height, image_width, self.height, self.width)

        return {"crop_coords": crop_coords}

