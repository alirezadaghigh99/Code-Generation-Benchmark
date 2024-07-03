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
        return "size", "scale", "ratio", "interpolation"class Crop(_BaseCrop):
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