class Downscale(ImageOnlyTransform):
    """Decreases image quality by downscaling and then upscaling it back to its original size.

    Args:
        scale_range (tuple[float, float]): A tuple defining the minimum and maximum scale to which the image
            will be downscaled. The range should be between 0 and 1, inclusive at minimum and exclusive at maximum.
            The first value should be less than or equal to the second value.
        interpolation_pair (InterpolationDict): A dictionary specifying the interpolation methods to use for
            downscaling and upscaling. Should include keys 'downscale' and 'upscale' with cv2 interpolation
                flags as values.
            Example: {"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_LINEAR}.

    Targets:
        image

    Image types:
        uint8, float32

    Example:
        >>> transform = Downscale(scale_range=(0.5, 0.9), interpolation_pair={"downscale": cv2.INTER_AREA,
                                                          "upscale": cv2.INTER_CUBIC})
        >>> transformed = transform(image=img)
    """

    class InitSchema(BaseTransformInitSchema):
        scale_min: float | None = Field(
            default=None,
            ge=0,
            le=1,
            description="Lower bound on the image scale.",
        )
        scale_max: float | None = Field(
            default=None,
            ge=0,
            lt=1,
            description="Upper bound on the image scale.",
        )

        interpolation: int | Interpolation | InterpolationDict | None = Field(
            default_factory=lambda: Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST),
        )
        interpolation_pair: InterpolationPydantic

        scale_range: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)] = (
            0.25,
            0.25,
        )

        @model_validator(mode="after")
        def validate_params(self) -> Self:
            if self.scale_min is not None and self.scale_max is not None:
                warn(
                    "scale_min and scale_max are deprecated. Use scale_range instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

                self.scale_range = (self.scale_min, self.scale_max)
                self.scale_min = None
                self.scale_max = None

            if self.interpolation is not None:
                warn(
                    "Downscale.interpolation is deprecated. Use Downscale.interpolation_pair instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

                if isinstance(self.interpolation, dict):
                    self.interpolation_pair = InterpolationPydantic(**self.interpolation)
                elif isinstance(self.interpolation, int):
                    self.interpolation_pair = InterpolationPydantic(
                        upscale=self.interpolation,
                        downscale=self.interpolation,
                    )
                elif isinstance(self.interpolation, Interpolation):
                    self.interpolation_pair = InterpolationPydantic(
                        upscale=self.interpolation.upscale,
                        downscale=self.interpolation.downscale,
                    )
                self.interpolation = None

            return self

    def __init__(
        self,
        scale_min: float | None = None,
        scale_max: float | None = None,
        interpolation: int | Interpolation | InterpolationDict | None = None,
        scale_range: tuple[float, float] = (0.25, 0.25),
        interpolation_pair: InterpolationDict = InterpolationDict(
            {"upscale": cv2.INTER_NEAREST, "downscale": cv2.INTER_NEAREST},
        ),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.scale_range = scale_range
        self.interpolation_pair = interpolation_pair

    def apply(self, img: np.ndarray, scale: float, **params: Any) -> np.ndarray:
        return fmain.downscale(
            img,
            scale=scale,
            down_interpolation=self.interpolation_pair["downscale"],
            up_interpolation=self.interpolation_pair["upscale"],
        )

    def get_params(self) -> dict[str, Any]:
        return {"scale": random.uniform(self.scale_range[0], self.scale_range[1])}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("scale_range", "interpolation_pair")

class TemplateTransform(ImageOnlyTransform):
    """Apply blending of input image with specified templates
    Args:
        templates (numpy array or list of numpy arrays): Images as template for transform.
        img_weight: If single float weight will be sampled from (0, img_weight).
            If tuple of float img_weight will be in range `[img_weight[0], img_weight[1])`.
            If you want fixed weight, use (img_weight, img_weight)
            Default: (0.5, 0.5).
        template_weight: If single float weight will be sampled from (0, template_weight).
            If tuple of float template_weight will be in range `[template_weight[0], template_weight[1])`.
            If you want fixed weight, use (template_weight, template_weight)
            Default: (0.5, 0.5).
        template_transform: transformation object which could be applied to template,
            must produce template the same size as input image.
        name: (Optional) Name of transform, used only for deserialization.
        p: probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        templates: np.ndarray | Sequence[np.ndarray] = Field(..., description="Images as template for transform.")
        img_weight: ZeroOneRangeType = (0.5, 0.5)
        template_weight: ZeroOneRangeType = (0.5, 0.5)
        template_transform: Callable[..., Any] | None = Field(
            default=None,
            description="Transformation object applied to template.",
        )
        name: str | None = Field(default=None, description="Name of transform, used only for deserialization.")

        @field_validator("templates")
        @classmethod
        def validate_templates(cls, v: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
            if isinstance(v, np.ndarray):
                return [v]
            if isinstance(v, list):
                if not all(isinstance(item, np.ndarray) for item in v):
                    msg = "All templates must be numpy arrays."
                    raise ValueError(msg)
                return v
            msg = "Templates must be a numpy array or a list of numpy arrays."
            raise TypeError(msg)

    def __init__(
        self,
        templates: np.ndarray | list[np.ndarray],
        img_weight: ScaleFloatType = (0.5, 0.5),
        template_weight: ScaleFloatType = (0.5, 0.5),
        template_transform: Callable[..., Any] | None = None,
        name: str | None = None,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.templates = templates
        self.img_weight = cast(Tuple[float, float], img_weight)
        self.template_weight = cast(Tuple[float, float], template_weight)
        self.template_transform = template_transform
        self.name = name

    def apply(
        self,
        img: np.ndarray,
        template: np.ndarray,
        img_weight: float,
        template_weight: float,
        **params: Any,
    ) -> np.ndarray:
        return add_weighted(img, img_weight, template, template_weight)

    def get_params(self) -> dict[str, float]:
        return {
            "img_weight": random.uniform(self.img_weight[0], self.img_weight[1]),
            "template_weight": random.uniform(self.template_weight[0], self.template_weight[1]),
        }

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        img = params["image"]
        template = random.choice(self.templates)

        if self.template_transform is not None:
            template = self.template_transform(image=template)["image"]

        if get_num_channels(template) not in [1, get_num_channels(img)]:
            msg = (
                "Template must be a single channel or "
                "has the same number of channels as input "
                f"image ({get_num_channels(img)}), got {get_num_channels(template)}"
            )
            raise ValueError(msg)

        if template.dtype != img.dtype:
            msg = "Image and template must be the same image type"
            raise ValueError(msg)

        if img.shape[:2] != template.shape[:2]:
            raise ValueError(f"Image and template must be the same size, got {img.shape[:2]} and {template.shape[:2]}")

        if get_num_channels(template) == 1 and get_num_channels(img) > 1:
            template = np.stack((template,) * get_num_channels(img), axis=-1)

        # in order to support grayscale image with dummy dim
        template = template.reshape(img.shape)

        return {"template": template}

    @classmethod
    def is_serializable(cls) -> bool:
        return False

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def to_dict_private(self) -> dict[str, Any]:
        if self.name is None:
            msg = (
                "To make a TemplateTransform serializable you should provide the `name` argument, "
                "e.g. `TemplateTransform(name='my_transform', ...)`."
            )
            raise ValueError(msg)
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}

class RandomFog(ImageOnlyTransform):
    """Simulates fog for the image.

    Args:
        fog_coef_range (tuple): tuple of bounds on the fog intensity coefficient (fog_coef_lower, fog_coef_upper).
            Default: (0.3, 1).
        alpha_coef (float): Transparency of the fog circles. Should be in [0, 1] range. Default: 0.08.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """

    class InitSchema(BaseTransformInitSchema):
        fog_coef_lower: float | None = Field(
            default=None,
            description="Lower limit for fog intensity coefficient",
            ge=0,
            le=1,
        )
        fog_coef_upper: float | None = Field(
            default=None,
            description="Upper limit for fog intensity coefficient",
            ge=0,
            le=1,
        )
        fog_coef_range: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)] = (
            0.3,
            1,
        )

        alpha_coef: float = Field(default=0.08, description="Transparency of the fog circles", ge=0, le=1)

        @model_validator(mode="after")
        def validate_fog_coefficients(self) -> Self:
            if self.fog_coef_lower is not None:
                warn("`fog_coef_lower` is deprecated, use `fog_coef_range` instead.", DeprecationWarning, stacklevel=2)
            if self.fog_coef_upper is not None:
                warn("`fog_coef_upper` is deprecated, use `fog_coef_range` instead.", DeprecationWarning, stacklevel=2)

            lower = self.fog_coef_lower if self.fog_coef_lower is not None else self.fog_coef_range[0]
            upper = self.fog_coef_upper if self.fog_coef_upper is not None else self.fog_coef_range[1]
            self.fog_coef_range = (lower, upper)

            self.fog_coef_lower = None
            self.fog_coef_upper = None

            return self

    def __init__(
        self,
        fog_coef_lower: float | None = None,
        fog_coef_upper: float | None = None,
        alpha_coef: float = 0.08,
        fog_coef_range: tuple[float, float] = (0.3, 1),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.fog_coef_range = fog_coef_range
        self.alpha_coef = alpha_coef

    def apply(
        self,
        img: np.ndarray,
        fog_coef: np.ndarray,
        haze_list: list[tuple[int, int]],
        **params: Any,
    ) -> np.ndarray:
        return fmain.add_fog(img, fog_coef, self.alpha_coef, haze_list)

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        img = params["image"]
        fog_coef = random.uniform(*self.fog_coef_range)

        height, width = imshape = img.shape[:2]

        hw = max(1, int(width // 3 * fog_coef))

        haze_list = []
        midx = width // 2 - 2 * hw
        midy = height // 2 - hw
        index = 1

        while midx > -hw or midy > -hw:
            for _ in range(hw // 10 * index):
                x = random.randint(midx, width - midx - hw)
                y = random.randint(midy, height - midy - hw)
                haze_list.append((x, y))

            midx -= 3 * hw * width // sum(imshape)
            midy -= 3 * hw * height // sum(imshape)
            index += 1

        return {"haze_list": haze_list, "fog_coef": fog_coef}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "fog_coef_range", "alpha_coef"

class RandomSunFlare(ImageOnlyTransform):
    """Simulates Sun Flare for the image

    Args:
        flare_roi (tuple[float, float, float, float]): Tuple specifying the region of the image where flare will
            appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
        src_radius (int): Radius of the source for the flare.
        src_color (tuple[int, int, int]): Color of the flare as an (R, G, B) tuple.
        angle_range (tuple[float, float]): tuple specifying the range of angles for the flare.
            Both ends of the range are in the [0, 1] interval.
        num_flare_circles_range (tuple[int, int]): tuple specifying the range for the number of flare circles.
        p (float): Probability of applying the transform.

    Targets:
        image

    Image types:
        uint8

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """

    class InitSchema(BaseTransformInitSchema):
        flare_roi: tuple[float, float, float, float] = Field(
            default=(0, 0, 1, 0.5),
            description="Region of the image where flare will appear",
        )
        angle_lower: float | None = Field(default=None, description="Lower bound for the angle", ge=0, le=1)
        angle_upper: float | None = Field(default=None, description="Upper bound for the angle", ge=0, le=1)

        num_flare_circles_lower: int | None = Field(
            default=6,
            description="Lower limit for the number of flare circles",
            ge=0,
        )
        num_flare_circles_upper: int | None = Field(
            default=10,
            description="Upper limit for the number of flare circles",
            gt=0,
        )
        src_radius: int = Field(default=400, description="Source radius for the flare")
        src_color: tuple[int, ...] = Field(default=(255, 255, 255), description="Color of the flare")

        angle_range: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)] = Field(
            default=(0, 1),
            description="Angle range",
        )

        num_flare_circles_range: Annotated[
            tuple[int, int],
            AfterValidator(check_1plus),
            AfterValidator(nondecreasing),
        ] = Field(default=(6, 10), description="Number of flare circles range")

        @model_validator(mode="after")
        def validate_parameters(self) -> Self:
            flare_center_lower_x, flare_center_lower_y, flare_center_upper_x, flare_center_upper_y = self.flare_roi
            if (
                not 0 <= flare_center_lower_x < flare_center_upper_x <= 1
                or not 0 <= flare_center_lower_y < flare_center_upper_y <= 1
            ):
                raise ValueError(f"Invalid flare_roi. Got: {self.flare_roi}")

            if self.angle_lower is not None or self.angle_upper is not None:
                if self.angle_lower is not None:
                    warn(
                        "`angle_lower` deprecated. Use `angle_range` as tuple (angle_lower, angle_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.angle_upper is not None:
                    warn(
                        "`angle_upper` deprecated. Use `angle_range` as tuple(angle_lower, angle_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = self.angle_lower if self.angle_lower is not None else self.angle_range[0]
                upper = self.angle_upper if self.angle_upper is not None else self.angle_range[1]
                self.angle_range = (lower, upper)

            if self.num_flare_circles_lower is not None or self.num_flare_circles_upper is not None:
                if self.num_flare_circles_lower is not None:
                    warn(
                        "`num_flare_circles_lower` deprecated. Use `num_flare_circles_range` as tuple"
                        " (num_flare_circles_lower, num_flare_circles_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.num_flare_circles_upper is not None:
                    warn(
                        "`num_flare_circles_upper` deprecated. Use `num_flare_circles_range` as tuple"
                        " (num_flare_circles_lower, num_flare_circles_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = (
                    self.num_flare_circles_lower
                    if self.num_flare_circles_lower is not None
                    else self.num_flare_circles_range[0]
                )
                upper = (
                    self.num_flare_circles_upper
                    if self.num_flare_circles_upper is not None
                    else self.num_flare_circles_range[1]
                )
                self.num_flare_circles_range = (lower, upper)

            return self

    def __init__(
        self,
        flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5),
        angle_lower: float | None = None,
        angle_upper: float | None = None,
        num_flare_circles_lower: int | None = None,
        num_flare_circles_upper: int | None = None,
        src_radius: int = 400,
        src_color: tuple[int, ...] = (255, 255, 255),
        angle_range: tuple[float, float] = (0, 1),
        num_flare_circles_range: tuple[int, int] = (6, 10),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.angle_range = angle_range
        self.num_flare_circles_range = num_flare_circles_range

        self.src_radius = src_radius
        self.src_color = src_color
        self.flare_roi = flare_roi

    def apply(
        self,
        img: np.ndarray,
        flare_center: tuple[float, float],
        circles: list[Any],
        **params: Any,
    ) -> np.ndarray:
        if circles is None:
            circles = []
        return fmain.add_sun_flare(
            img,
            flare_center,
            self.src_radius,
            self.src_color,
            circles,
        )

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        img = params["image"]
        height, width = img.shape[:2]

        angle = 2 * math.pi * random.uniform(*self.angle_range)

        (flare_center_lower_x, flare_center_lower_y, flare_center_upper_x, flare_center_upper_y) = self.flare_roi

        flare_center_x = random.uniform(flare_center_lower_x, flare_center_upper_x)
        flare_center_y = random.uniform(flare_center_lower_y, flare_center_upper_y)

        flare_center_x = int(width * flare_center_x)
        flare_center_y = int(height * flare_center_y)

        num_circles = random.randint(*self.num_flare_circles_range)

        circles = []

        x = []
        y = []

        def line(t: float) -> tuple[float, float]:
            return (flare_center_x + t * math.cos(angle), flare_center_y + t * math.sin(angle))

        for t_val in range(-flare_center_x, width - flare_center_x, 10):
            rand_x, rand_y = line(t_val)
            x.append(rand_x)
            y.append(rand_y)

        for _ in range(num_circles):
            alpha = random.uniform(0.05, 0.2)
            r = random.randint(0, len(x) - 1)
            rad = random.randint(1, max(height // 100 - 2, 2))

            r_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            g_color = random.randint(max(self.src_color[1] - 50, 0), self.src_color[1])
            b_color = random.randint(max(self.src_color[2] - 50, 0), self.src_color[2])

            circles += [
                (
                    alpha,
                    (int(x[r]), int(y[r])),
                    pow(rad, 3),
                    (r_color, g_color, b_color),
                ),
            ]

        return {
            "circles": circles,
            "flare_center": (flare_center_x, flare_center_y),
        }

    def get_transform_init_args(self) -> dict[str, Any]:
        return {
            "flare_roi": self.flare_roi,
            "angle_range": self.angle_range,
            "num_flare_circles_range": self.num_flare_circles_range,
            "src_radius": self.src_radius,
            "src_color": self.src_color,
        }

class MultiplicativeNoise(ImageOnlyTransform):
    """Multiply image by a random number or array of numbers.

    Args:
        multiplier: If a single float, the image will be multiplied by this number.
            If a tuple of floats, the multiplier will be a random number in the range `[multiplier[0], multiplier[1])`.
            Default: (0.9, 1.1).
        elementwise: If `False`, multiply all pixels in the image by a single random value sampled once.
            If `True`, multiply image pixels by values that are pixelwise randomly sampled. Default: False.
        p: Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, np.float32

    """

    class InitSchema(BaseTransformInitSchema):
        multiplier: Annotated[tuple[float, float], AfterValidator(check_0plus), AfterValidator(nondecreasing)] = (
            0.9,
            1.1,
        )
        per_channel: bool | None = Field(
            default=False,
            description="Apply multiplier per channel.",
            deprecated="Does not have any effect. Will be removed in future releases.",
        )
        elementwise: bool = Field(default=False, description="Apply multiplier element-wise to pixels.")

    def __init__(
        self,
        multiplier: ScaleFloatType = (0.9, 1.1),
        per_channel: bool | None = None,
        elementwise: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.multiplier = cast(Tuple[float, float], multiplier)
        self.elementwise = elementwise

    def apply(
        self,
        img: np.ndarray,
        multiplier: float | np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return multiply(img, multiplier)

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.multiplier[0] == self.multiplier[1]:
            return {"multiplier": self.multiplier[0]}

        img = params["image"]

        num_channels = get_num_channels(img)

        shape = img.shape if self.elementwise else [num_channels]

        multiplier = random_utils.uniform(self.multiplier[0], self.multiplier[1], shape).astype(np.float32)

        return {"multiplier": multiplier}

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "multiplier", "elementwise"

class ImageCompression(ImageOnlyTransform):
    """Decreases image quality by Jpeg, WebP compression of an image.

    Args:
        quality_range: tuple of bounds on the image quality i.e. (quality_lower, quality_upper).
            Both values should be in [1, 100] range.
        compression_type (ImageCompressionType): should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
            Default: ImageCompressionType.JPEG

    Targets:
        image

    Image types:
        uint8, float32

    """

    class InitSchema(BaseTransformInitSchema):
        quality_range: Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)] = (
            99,
            100,
        )

        quality_lower: int | None = Field(
            default=None,
            description="Lower bound on the image quality",
            ge=1,
            le=100,
        )
        quality_upper: int | None = Field(
            default=None,
            description="Upper bound on the image quality",
            ge=1,
            le=100,
        )
        compression_type: ImageCompressionType = Field(
            default=ImageCompressionType.JPEG,
            description="Image compression format",
        )

        @model_validator(mode="after")
        def validate_ranges(self) -> Self:
            # Update the quality_range based on the non-None values of quality_lower and quality_upper
            if self.quality_lower is not None or self.quality_upper is not None:
                if self.quality_lower is not None:
                    warn(
                        "`quality_lower` is deprecated. Use `quality_range` as tuple"
                        " (quality_lower, quality_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.quality_upper is not None:
                    warn(
                        "`quality_upper` is deprecated. Use `quality_range` as tuple"
                        " (quality_lower, quality_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = self.quality_lower if self.quality_lower is not None else self.quality_range[0]
                upper = self.quality_upper if self.quality_upper is not None else self.quality_range[1]
                self.quality_range = (lower, upper)
                # Clear the deprecated individual quality settings
                self.quality_lower = None
                self.quality_upper = None

            # Validate the quality_range
            if not (1 <= self.quality_range[0] <= MAX_JPEG_QUALITY and 1 <= self.quality_range[1] <= MAX_JPEG_QUALITY):
                raise ValueError(f"Quality range values should be within [1, {MAX_JPEG_QUALITY}] range.")

            return self

    def __init__(
        self,
        quality_lower: int | None = None,
        quality_upper: int | None = None,
        compression_type: ImageCompressionType = ImageCompressionType.JPEG,
        quality_range: tuple[int, int] = (99, 100),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)
        self.quality_range = quality_range
        self.compression_type = compression_type

    def apply(self, img: np.ndarray, quality: int, image_type: Literal[".jpg", ".webp"], **params: Any) -> np.ndarray:
        if img.ndim != MONO_CHANNEL_DIMENSIONS and img.shape[-1] not in (1, 3, 4):
            msg = "ImageCompression transformation expects 1, 3 or 4 channel images."
            raise TypeError(msg)
        return fmain.image_compression(img, quality, image_type)

    def get_params(self) -> dict[str, int | str]:
        if self.compression_type == ImageCompressionType.JPEG:
            image_type = ".jpg"
        elif self.compression_type == ImageCompressionType.WEBP:
            image_type = ".webp"
        else:
            raise ValueError(f"Unknown image compression type: {self.compression_type}")

        return {
            "quality": random.randint(self.quality_range[0], self.quality_range[1]),
            "image_type": image_type,
        }

    def get_transform_init_args(self) -> dict[str, Any]:
        return {
            "quality_range": self.quality_range,
            "compression_type": self.compression_type.value,
        }

class RandomRain(ImageOnlyTransform):
    """Adds rain effects to an image.

    Args:
        slant_range (tuple[int, int]): tuple of type (slant_lower, slant_upper) representing the range for
            rain slant angle.
        drop_length (int): Length of the raindrops.
        drop_width (int): Width of the raindrops.
        drop_color (tuple[int, int, int]): Color of the rain drops in RGB format.
        blur_value (int): Blur value for simulating rain effect. Rainy views are blurry.
        brightness_coefficient (float): Coefficient to adjust the brightness of the image.
            Rainy days are usually shady. Should be in the range (0, 1].
        rain_type (Optional[str]): Type of rain to simulate. One of [None, "drizzle", "heavy", "torrential"].


    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """

    class InitSchema(BaseTransformInitSchema):
        slant_lower: int | None = Field(
            default=None,
            description="Lower bound for rain slant angle",
        )
        slant_upper: int | None = Field(
            default=None,
            description="Upper bound for rain slant angle",
        )
        slant_range: Annotated[tuple[float, float], AfterValidator(nondecreasing)] = Field(
            default=(-10, 10),
            description="tuple like (slant_lower, slant_upper) for rain slant angle",
        )
        drop_length: int = Field(default=20, description="Length of raindrops", ge=1)
        drop_width: int = Field(default=1, description="Width of raindrops", ge=1)
        drop_color: tuple[int, int, int] = Field(default=(200, 200, 200), description="Color of raindrops")
        blur_value: int = Field(default=7, description="Blur value for simulating rain effect", ge=1)
        brightness_coefficient: float = Field(
            default=0.7,
            description="Brightness coefficient for rainy effect",
            gt=0,
            le=1,
        )
        rain_type: RainMode | None = Field(default=None, description="Type of rain to simulate")

        @model_validator(mode="after")
        def validate_ranges(self) -> Self:
            if self.slant_lower is not None or self.slant_upper is not None:
                if self.slant_lower is not None:
                    warn(
                        "`slant_lower` deprecated. Use `slant_range` as tuple (slant_lower, slant_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.slant_upper is not None:
                    warn(
                        "`slant_upper` deprecated. Use `slant_range` as tuple (slant_lower, slant_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = self.slant_lower if self.slant_lower is not None else self.slant_range[0]
                upper = self.slant_upper if self.slant_upper is not None else self.slant_range[1]
                self.slant_range = (lower, upper)
                self.slant_lower = None
                self.slant_upper = None

            # Validate the slant_range
            if not (-MAX_RAIN_ANGLE <= self.slant_range[0] <= self.slant_range[1] <= MAX_RAIN_ANGLE):
                raise ValueError(
                    f"slant_range values should be increasing within [-{MAX_RAIN_ANGLE}, {MAX_RAIN_ANGLE}] range.",
                )
            return self

    def __init__(
        self,
        slant_lower: int | None = None,
        slant_upper: int | None = None,
        slant_range: tuple[int, int] = (-10, 10),
        drop_length: int = 20,
        drop_width: int = 1,
        drop_color: tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: RainMode | None = None,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.slant_range = slant_range
        self.drop_length = drop_length
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.rain_type = rain_type

    def apply(
        self,
        img: np.ndarray,
        slant: int,
        drop_length: int,
        rain_drops: list[tuple[int, int]],
        **params: Any,
    ) -> np.ndarray:
        return fmain.add_rain(
            img,
            slant,
            drop_length,
            self.drop_width,
            self.drop_color,
            self.blur_value,
            self.brightness_coefficient,
            rain_drops,
        )

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        img = params["image"]
        slant = int(random.uniform(*self.slant_range))

        height, width = img.shape[:2]
        area = height * width

        if self.rain_type == "drizzle":
            num_drops = area // 770
            drop_length = 10
        elif self.rain_type == "heavy":
            num_drops = width * height // 600
            drop_length = 30
        elif self.rain_type == "torrential":
            num_drops = area // 500
            drop_length = 60
        else:
            drop_length = self.drop_length
            num_drops = area // 600

        rain_drops = []

        for _ in range(num_drops):  # If You want heavy rain, try increasing this
            x = random.randint(slant, width) if slant < 0 else random.randint(0, width - slant)

            y = random.randint(0, height - drop_length)

            rain_drops.append((x, y))

        return {"drop_length": drop_length, "slant": slant, "rain_drops": rain_drops}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "slant_range",
            "drop_length",
            "drop_width",
            "drop_color",
            "blur_value",
            "brightness_coefficient",
            "rain_type",
        )

class RandomSnow(ImageOnlyTransform):
    """Bleach out some pixel values imitating snow.

    Args:
        snow_point_range (tuple): tuple of bounds on the amount of snow i.e. (snow_point_lower, snow_point_upper).
            Both values should be in the (0, 1) range. Default: (0.1, 0.3).
        brightness_coeff (float): Coefficient applied to increase the brightness of pixels
            below the snow_point threshold. Larger values lead to more pronounced snow effects.
            Should be > 0. Default: 2.5.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """

    class InitSchema(BaseTransformInitSchema):
        snow_point_range: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)] = (
            Field(
                default=(0.1, 0.3),
                description="lower and upper bound on the amount of snow as tuple (snow_point_lower, snow_point_upper)",
            )
        )
        snow_point_lower: float | None = Field(
            default=None,
            description="Lower bound of the amount of snow",
            gt=0,
            lt=1,
        )
        snow_point_upper: float | None = Field(
            default=None,
            description="Upper bound of the amount of snow",
            gt=0,
            lt=1,
        )
        brightness_coeff: float = Field(default=2.5, description="Brightness coefficient, must be > 0", gt=0)

        @model_validator(mode="after")
        def validate_ranges(self) -> Self:
            if self.snow_point_lower is not None or self.snow_point_upper is not None:
                if self.snow_point_lower is not None:
                    warn(
                        "`snow_point_lower` deprecated. Use `snow_point_range` as tuple"
                        " (snow_point_lower, snow_point_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.snow_point_upper is not None:
                    warn(
                        "`snow_point_upper` deprecated. Use `snow_point_range` as tuple"
                        "(snow_point_lower, snow_point_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = self.snow_point_lower if self.snow_point_lower is not None else self.snow_point_range[0]
                upper = self.snow_point_upper if self.snow_point_upper is not None else self.snow_point_range[1]
                self.snow_point_range = (lower, upper)
                self.snow_point_lower = None
                self.snow_point_upper = None

            # Validate the snow_point_range
            if not (0 < self.snow_point_range[0] <= self.snow_point_range[1] < 1):
                raise ValueError("snow_point_range values should be increasing within (0, 1) range.")

            return self

    def __init__(
        self,
        snow_point_lower: float | None = None,
        snow_point_upper: float | None = None,
        brightness_coeff: float = 2.5,
        snow_point_range: tuple[float, float] = (0.1, 0.3),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)

        self.snow_point_range = snow_point_range
        self.brightness_coeff = brightness_coeff

    def apply(self, img: np.ndarray, snow_point: float, **params: Any) -> np.ndarray:
        return fmain.add_snow(img, snow_point, self.brightness_coeff)

    def get_params(self) -> dict[str, np.ndarray]:
        return {"snow_point": random.uniform(*self.snow_point_range)}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "snow_point_range", "brightness_coeff"

class UnsharpMask(ImageOnlyTransform):
    """Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.

    Args:
        blur_limit: maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit: Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        alpha: range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Default: (0.2, 0.5).
        threshold: Value to limit sharpening only for areas with high pixel difference between original image
            and it's smoothed version. Higher threshold means less sharpening on flat areas.
            Must be in range [0, 255]. Default: 10.
        p: probability of applying the transform. Default: 0.5.

    Reference:
        arxiv.org/pdf/2107.10833.pdf

    Targets:
        image

    """

    class InitSchema(BaseTransformInitSchema):
        sigma_limit: NonNegativeFloatRangeType = 0
        alpha: ZeroOneRangeType = (0.2, 0.5)
        threshold: int = Field(default=10, ge=0, le=255, description="Threshold for limiting sharpening.")

        blur_limit: ScaleIntType = Field(
            default=(3, 7),
            description="Maximum kernel size for blurring the input image.",
        )

        @field_validator("blur_limit")
        @classmethod
        def process_blur(cls, value: ScaleIntType, info: ValidationInfo) -> tuple[int, int]:
            return process_blur_limit(value, info, min_value=3)

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigma_limit: ScaleFloatType = 0.0,
        alpha: ScaleFloatType = (0.2, 0.5),
        threshold: int = 10,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.blur_limit = cast(Tuple[int, int], blur_limit)
        self.sigma_limit = cast(Tuple[float, float], sigma_limit)
        self.alpha = cast(Tuple[float, float], alpha)
        self.threshold = threshold

    def get_params(self) -> dict[str, Any]:
        return {
            "ksize": random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2),
            "sigma": random.uniform(*self.sigma_limit),
            "alpha": random.uniform(*self.alpha),
        }

    def apply(self, img: np.ndarray, ksize: int, sigma: int, alpha: float, **params: Any) -> np.ndarray:
        return fmain.unsharp_mask(img, ksize, sigma=sigma, alpha=alpha, threshold=self.threshold)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "blur_limit", "sigma_limit", "alpha", "threshold"

class RandomShadow(ImageOnlyTransform):
    """Simulates shadows for the image

    Args:
        shadow_roi: region of the image where shadows
            will appear. All values should be in range [0, 1].
        num_shadows_limit: Lower and upper limits for the possible number of shadows.
        shadow_dimension: number of edges in the shadow polygons

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """

    class InitSchema(BaseTransformInitSchema):
        shadow_roi: tuple[float, float, float, float] = Field(
            default=(0, 0.5, 1, 1),
            description="Region of the image where shadows will appear",
        )
        num_shadows_limit: Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)] = (
            1,
            2,
        )
        num_shadows_lower: int | None = Field(
            default=None,
            description="Lower limit for the possible number of shadows",
        )
        num_shadows_upper: int | None = Field(
            default=None,
            description="Upper limit for the possible number of shadows",
        )
        shadow_dimension: int = Field(default=5, description="Number of edges in the shadow polygons", ge=1)

        @model_validator(mode="after")
        def validate_shadows(self) -> Self:
            if self.num_shadows_lower is not None:
                warn(
                    "`num_shadows_lower` is deprecated. Use `num_shadows_limit` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if self.num_shadows_upper is not None:
                warn(
                    "`num_shadows_upper` is deprecated. Use `num_shadows_limit` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if self.num_shadows_lower is not None or self.num_shadows_upper is not None:
                num_shadows_lower = (
                    self.num_shadows_lower if self.num_shadows_lower is not None else self.num_shadows_limit[0]
                )
                num_shadows_upper = (
                    self.num_shadows_upper if self.num_shadows_upper is not None else self.num_shadows_limit[1]
                )

                self.num_shadows_limit = (num_shadows_lower, num_shadows_upper)
                self.num_shadows_lower = None
                self.num_shadows_upper = None

            shadow_lower_x, shadow_lower_y, shadow_upper_x, shadow_upper_y = self.shadow_roi

            if not 0 <= shadow_lower_x <= shadow_upper_x <= 1 or not 0 <= shadow_lower_y <= shadow_upper_y <= 1:
                raise ValueError(f"Invalid shadow_roi. Got: {self.shadow_roi}")

            return self

    def __init__(
        self,
        shadow_roi: tuple[float, float, float, float] = (0, 0.5, 1, 1),
        num_shadows_limit: tuple[int, int] = (1, 2),
        num_shadows_lower: int | None = None,
        num_shadows_upper: int | None = None,
        shadow_dimension: int = 5,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.shadow_roi = shadow_roi
        self.shadow_dimension = shadow_dimension
        self.num_shadows_limit = num_shadows_limit

    def apply(self, img: np.ndarray, vertices_list: list[np.ndarray], **params: Any) -> np.ndarray:
        return fmain.add_shadow(img, vertices_list)

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, list[np.ndarray]]:
        img = params["image"]
        height, width = img.shape[:2]

        num_shadows = random.randint(self.num_shadows_limit[0], self.num_shadows_limit[1])

        x_min, y_min, x_max, y_max = self.shadow_roi

        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        vertices_list = [
            np.stack(
                [
                    random_utils.randint(x_min, x_max, size=5),
                    random_utils.randint(y_min, y_max, size=5),
                ],
                axis=1,
            )
            for _ in range(num_shadows)
        ]

        return {"vertices_list": vertices_list}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "shadow_roi",
            "num_shadows_limit",
            "shadow_dimension",
        )

class ColorJitter(ImageOnlyTransform):
    """Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
    this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
    Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
    overflow, but we use value saturation.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            If float:
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            If tuple[float, float]] will be sampled from that range. Both values should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            If float:
                contrast_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            If tuple[float, float]] will be sampled from that range. Both values should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            If float:
               saturation_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            If tuple[float, float]] will be sampled from that range. Both values should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            If float:
               saturation_factor is chosen uniformly from [-hue, hue]. Should have 0 <= hue <= 0.5.
            If tuple[float, float]] will be sampled from that range. Both values should be in range [-0.5, 0.5].

    """

    class InitSchema(BaseTransformInitSchema):
        brightness: Annotated[ScaleFloatType, Field(default=0.2, description="Range for jittering brightness.")]
        contrast: Annotated[ScaleFloatType, Field(default=0.2, description="Range for jittering contrast.")]
        saturation: Annotated[ScaleFloatType, Field(default=0.2, description="Range for jittering saturation.")]
        hue: Annotated[ScaleFloatType, Field(default=0.2, description="Range for jittering hue.")]

        @field_validator("brightness", "contrast", "saturation", "hue")
        @classmethod
        def check_ranges(cls, value: ScaleFloatType, info: ValidationInfo) -> tuple[float, float]:
            if info.field_name == "hue":
                bounds = -0.5, 0.5
                bias = 0
                clip = False
            elif info.field_name in ["brightness", "contrast", "saturation"]:
                bounds = 0, float("inf")
                bias = 1
                clip = True

            if isinstance(value, numbers.Number):
                if value < 0:
                    raise ValueError(f"If {info.field_name} is a single number, it must be non negative.")
                value = [bias - value, bias + value]
                if clip:
                    value[0] = max(value[0], 0)
            elif isinstance(value, (tuple, list)) and len(value) == PAIR:
                check_range(value, *bounds, info.field_name)

            return cast(Tuple[float, float], value)

    def __init__(
        self,
        brightness: ScaleFloatType = (0.8, 1),
        contrast: ScaleFloatType = (0.8, 1),
        saturation: ScaleFloatType = (0.8, 1),
        hue: ScaleFloatType = (-0.5, 0.5),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.brightness = cast(Tuple[float, float], brightness)
        self.contrast = cast(Tuple[float, float], contrast)
        self.saturation = cast(Tuple[float, float], saturation)
        self.hue = cast(Tuple[float, float], hue)

        self.transforms = [
            fmain.adjust_brightness_torchvision,
            fmain.adjust_contrast_torchvision,
            fmain.adjust_saturation_torchvision,
            fmain.adjust_hue_torchvision,
        ]

    def get_params(self) -> dict[str, Any]:
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])
        hue = random.uniform(self.hue[0], self.hue[1])

        order = [0, 1, 2, 3]
        order = random_utils.shuffle(order)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "order": order,
        }

    def apply(
        self,
        img: np.ndarray,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        order: list[int],
        **params: Any,
    ) -> np.ndarray:
        if order is None:
            order = [0, 1, 2, 3]
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "ColorJitter transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)
        color_transforms = [brightness, contrast, saturation, hue]
        for i in order:
            img = self.transforms[i](img, color_transforms[i])
        return img

    def get_transform_init_args_names(self) -> tuple[str, str, str, str]:
        return ("brightness", "contrast", "saturation", "hue")

class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

    Args:
        hue_shift_limit: range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        sat_shift_limit: range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        val_shift_limit: range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    class InitSchema(BaseTransformInitSchema):
        hue_shift_limit: SymmetricRangeType = (-20, 20)
        sat_shift_limit: SymmetricRangeType = (-30, 30)
        val_shift_limit: SymmetricRangeType = (-20, 20)

    def __init__(
        self,
        hue_shift_limit: ScaleIntType = 20,
        sat_shift_limit: ScaleIntType = 30,
        val_shift_limit: ScaleIntType = 20,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.hue_shift_limit = cast(Tuple[float, float], hue_shift_limit)
        self.sat_shift_limit = cast(Tuple[float, float], sat_shift_limit)
        self.val_shift_limit = cast(Tuple[float, float], val_shift_limit)

    def apply(
        self,
        img: np.ndarray,
        hue_shift: int,
        sat_shift: int,
        val_shift: int,
        **params: Any,
    ) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "HueSaturationValue transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)
        return fmain.shift_hsv(img, hue_shift, sat_shift, val_shift)

    def get_params(self) -> dict[str, float]:
        return {
            "hue_shift": random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
            "sat_shift": random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
            "val_shift": random.uniform(self.val_shift_limit[0], self.val_shift_limit[1]),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")

class Equalize(ImageOnlyTransform):
    """Equalize the image histogram.

    Args:
        mode (str): {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable.
            Function signature must include `image` argument.
        mask_params (list of str): Params for mask function.

    Targets:
        image

    Image types:
        uint8

    """

    class InitSchema(BaseTransformInitSchema):
        mode: ImageMode = "cv"
        by_channels: Annotated[bool, Field(default=True, description="Equalize channels separately if True")]
        mask: Annotated[
            np.ndarray | Callable[..., Any] | None,
            Field(default=None, description="Mask to apply for equalization"),
        ]
        mask_params: Annotated[Sequence[str], Field(default=[], description="Parameters for mask function")]

    def __init__(
        self,
        mode: ImageMode = "cv",
        by_channels: bool = True,
        mask: np.ndarray | Callable[..., Any] | None = None,
        mask_params: Sequence[str] = (),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, img: np.ndarray, mask: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.equalize(img, mode=self.mode, by_channels=self.by_channels, mask=mask)

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        if not callable(self.mask):
            return {"mask": self.mask}

        return {"mask": self.mask(**params)}

    @property
    def targets_as_params(self) -> list[str]:
        return ["image", *list(self.mask_params)]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("mode", "by_channels", "mask", "mask_params")

class RandomGridShuffle(DualTransform):
    """Randomly shuffles the grid's cells on an image, mask, or keypoints,
    effectively rearranging patches within the image.
    This transformation divides the image into a grid and then permutes these grid cells based on a random mapping.


    Args:
        grid (tuple[int, int]): Size of the grid for splitting the image into cells. Each cell is shuffled randomly.
        p (float): Probability that the transform will be applied.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32

    Examples:
        >>> import albumentations as A
        >>> transform = A.Compose([
            A.RandomGridShuffle(grid=(3, 3), p=1.0)
        ])
        >>> transformed = transform(image=my_image, mask=my_mask)
        >>> image, mask = transformed['image'], transformed['mask']
        # This will shuffle the 3x3 grid cells of `my_image` and `my_mask` randomly.
        # Mask and image are shuffled in a consistent way
    Note:
        This transform could be useful when only micro features are important for the model, and memorizing
        the global structure could be harmful. For example:
        - Identifying the type of cell phone used to take a picture based on micro artifacts generated by
        phone post-processing algorithms, rather than the semantic features of the photo.
        See more at https://ieeexplore.ieee.org/abstract/document/8622031
        - Identifying stress, glucose, hydration levels based on skin images.
    """

    class InitSchema(BaseTransformInitSchema):
        grid: OnePlusIntRangeType = (3, 3)

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS)

    def __init__(self, grid: tuple[int, int] = (3, 3), p: float = 0.5, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        self.grid = grid

    def apply(self, img: np.ndarray, tiles: np.ndarray, mapping: list[int], **params: Any) -> np.ndarray:
        return fmain.swap_tiles_on_image(img, tiles, mapping)

    def apply_to_mask(self, mask: np.ndarray, tiles: np.ndarray, mapping: list[int], **params: Any) -> np.ndarray:
        return fmain.swap_tiles_on_image(mask, tiles, mapping)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        tiles: np.ndarray,
        mapping: list[int],
        **params: Any,
    ) -> KeypointInternalType:
        x, y = keypoint[:2]

        # Find which original tile the keypoint belongs to
        for original_index, new_index in enumerate(mapping):
            start_y, start_x, end_y, end_x = tiles[original_index]
            # check if the keypoint is in this tile
            if start_y <= y < end_y and start_x <= x < end_x:
                # Get the new tile's coordinates
                new_start_y, new_start_x = tiles[new_index][:2]

                # Map the keypoint to the new tile's position
                new_x = (x - start_x) + new_start_x
                new_y = (y - start_y) + new_start_y

                return (new_x, new_y, *keypoint[2:])

        # If the keypoint wasn't in any tile (shouldn't happen), log a warning for debugging purposes
        warn(
            "Keypoint not in any tile, returning it unchanged. This is unexpected and should be investigated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return keypoint

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        height, width = params["image"].shape[:2]
        random_state = random_utils.get_random_state()
        original_tiles = fmain.split_uniform_grid(
            (height, width),
            self.grid,
            random_state=random_state,
        )
        shape_groups = fmain.create_shape_groups(original_tiles)
        mapping = fmain.shuffle_tiles_within_shape_groups(shape_groups, random_state=random_state)

        return {"tiles": original_tiles, "mapping": mapping}

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("grid",)

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "keypoints": self.apply_to_keypoints,
        }

class GaussNoise(ImageOnlyTransform):
    """Apply Gaussian noise to the input image.

    Args:
        var_limit (Union[float, tuple[float, float]]): Variance range for noise.
            If var_limit is a single float, the range will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): Mean of the noise. Default: 0
        per_channel (bool): If set to True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels.
            Faster when `per_channel = False`.
            Default: True
        noise_scale_factor (float): Scaling factor for noise generation. Value should be in the range (0, 1].
            When set to 1, noise is sampled for each pixel independently. If less, noise is sampled for a smaller size
            and resized to fit the shape of the image. Smaller values make the transform faster. Default: 1.0.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    class InitSchema(BaseTransformInitSchema):
        var_limit: NonNegativeFloatRangeType = Field(default=(10.0, 50.0), description="Variance range for noise.")
        mean: float = Field(default=0, description="Mean of the noise.")
        per_channel: bool = Field(default=True, description="Apply noise per channel.")
        noise_scale_factor: float = Field(gt=0, le=1)

    def __init__(
        self,
        var_limit: ScaleFloatType = (10.0, 50.0),
        mean: float = 0,
        per_channel: bool = True,
        noise_scale_factor: float = 1,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.var_limit = cast(Tuple[float, float], var_limit)
        self.mean = mean
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor

    def apply(self, img: np.ndarray, gauss: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.add_noise(img, gauss)

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, float]:
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = math.sqrt(var)

        if self.per_channel:
            target_shape = image.shape
            if self.noise_scale_factor == 1:
                gauss = random_utils.normal(self.mean, sigma, target_shape)
            else:
                gauss = fmain.generate_approx_gaussian_noise(target_shape, self.mean, sigma, self.noise_scale_factor)
        else:
            target_shape = image.shape[:2]
            if self.noise_scale_factor == 1:
                gauss = random_utils.normal(self.mean, sigma, target_shape)
            else:
                gauss = fmain.generate_approx_gaussian_noise(target_shape, self.mean, sigma, self.noise_scale_factor)

            if image.ndim > MONO_CHANNEL_DIMENSIONS:
                gauss = np.expand_dims(gauss, -1)

        return {"gauss": gauss}

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "var_limit", "per_channel", "mean", "noise_scale_factor"

class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearrange channels of the image.

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def apply(self, img: np.ndarray, channels_shuffled: tuple[int, ...], **params: Any) -> np.ndarray:
        return fmain.channel_shuffle(img, channels_shuffled)

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        img = params["image"]
        ch_arr = list(range(img.shape[2]))
        ch_arr = random_utils.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()

