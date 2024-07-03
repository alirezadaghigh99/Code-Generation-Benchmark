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
        return "fog_coef_range", "alpha_coef"