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
        return ("scale_range", "interpolation_pair")class TemplateTransform(ImageOnlyTransform):
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
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}class RandomFog(ImageOnlyTransform):
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
        return "fog_coef_range", "alpha_coef"class MultiplicativeNoise(ImageOnlyTransform):
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
        return "multiplier", "elementwise"class ImageCompression(ImageOnlyTransform):
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
        }class RandomRain(ImageOnlyTransform):
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
        )class RandomSnow(ImageOnlyTransform):
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