class Affine(DualTransform):
    """Augmentation to apply affine transformations to images.

    Affine transformations involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a defined content, e.g.
    if the image is translated to the left, pixels are created on the right.
    A method has to be defined to deal with these pixel values.
    The parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameters `interpolation` and
    `mask_interpolation` deals with the method of interpolation used for this.

    Args:
        scale (number, tuple of number or dict): Scaling factor to use, where ``1.0`` denotes "no change" and
            ``0.5`` is zoomed out to ``50`` percent of the original size.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That the same range will be used for both x- and y-axis. To keep the aspect ratio, set
                  ``keep_ratio=True``, then the same value will be used for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes. Note that when
                  the ``keep_ratio=True``, the x- and y-axis ranges should be the same.
        translate_percent (None, number, tuple of number or dict): Translation as a fraction of the image height/width
            (x-translation, y-translation), where ``0`` denotes "no change"
            and ``0.5`` denotes "half of the axis size".
                * If ``None`` then equivalent to ``0.0`` unless `translate_px` has a value other than ``None``.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That sampled fraction value will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        translate_px (None, int, tuple of int or dict): Translation in pixels.
                * If ``None`` then equivalent to ``0`` unless `translate_percent` has a value other than ``None``.
                * If a single int, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from
                  the discrete interval ``[a..b]``. That number will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        rotate (number or tuple of number): Rotation in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``. Rotation happens around the *center* of the image,
            not the top left corner as in some other frameworks.
                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``
                  and used as the rotation value.
        shear (number, tuple of number or dict): Shear in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``, with reasonable values being in the range of ``[-45, 45]``.
                * If a number, then that value will be used for all images as
                  the shear on the x-axis (no shear on the y-axis will be done).
                * If a tuple ``(a, b)``, then two value will be uniformly sampled per image
                  from the interval ``[a, b]`` and be used as the x- and y-shear value.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        interpolation (int): OpenCV interpolation flag.
        mask_interpolation (int): OpenCV interpolation flag.
        cval (number or sequence of number): The constant value to use when filling in newly created pixels.
            (E.g. translating by 1px to the right will create a new 1px-wide column of pixels
            on the left of the image).
            The value is only used when `mode=constant`. The expected value range is ``[0, 255]`` for ``uint8`` images.
        cval_mask (number or tuple of number): Same as cval but only for masks.
        mode (int): OpenCV border flag.
        fit_output (bool): If True, the image plane size and position will be adjusted to tightly capture
            the whole image after affine transformation (`translate_percent` and `translate_px` are ignored).
            Otherwise (``False``),  parts of the transformed image may end up outside the image plane.
            Fitting the output shape can be useful to avoid corners of the image being outside the image plane
            after applying rotations. Default: False
        keep_ratio (bool): When True, the original aspect ratio will be kept when the random scale is applied.
            Default: False.
        rotate_method (Literal["largest_box", "ellipse"]): rotation method used for the bounding boxes.
            Should be one of "largest_box" or "ellipse"[1]. Default: "largest_box"
        balanced_scale (bool): When True, scaling factors are chosen to be either entirely below or above 1,
            ensuring balanced scaling. Default: False.

            This is important because without it, scaling tends to lean towards upscaling. For example, if we want
            the image to zoom in and out by 2x, we may pick an interval [0.5, 2]. Since the interval [0.5, 1] is
            three times smaller than [1, 2], values above 1 are picked three times more often if sampled directly
            from [0.5, 2]. With `balanced_scale`, the  function ensures that half the time, the scaling
            factor is picked from below 1 (zooming out), and the other half from above 1 (zooming in).
            This makes the zooming in and out process more balanced.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    Reference:
        [1] https://arxiv.org/abs/2109.13488

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        scale: ScaleFloatType | dict[str, Any] | None = Field(
            default=None,
            description="Scaling factor or dictionary for independent axis scaling.",
        )
        translate_percent: ScaleFloatType | dict[str, Any] | None = Field(
            default=None,
            description="Translation as a fraction of the image dimension.",
        )
        translate_px: ScaleIntType | dict[str, Any] | None = Field(
            default=None,
            description="Translation in pixels.",
        )
        rotate: ScaleFloatType | None = Field(default=None, description="Rotation angle in degrees.")
        shear: ScaleFloatType | dict[str, Any] | None = Field(
            default=None,
            description="Shear angle in degrees.",
        )
        interpolation: InterpolationType = cv2.INTER_LINEAR
        mask_interpolation: InterpolationType = cv2.INTER_NEAREST

        cval: ColorType = Field(default=0, description="Value used for constant padding.")
        cval_mask: ColorType = Field(default=0, description="Value used for mask constant padding.")
        mode: BorderModeType = cv2.BORDER_CONSTANT
        fit_output: Annotated[bool, Field(default=False, description="Adjust output to capture whole image.")]
        keep_ratio: Annotated[bool, Field(default=False, description="Maintain aspect ratio when scaling.")]
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box"
        balanced_scale: Annotated[bool, Field(default=False, description="Use balanced scaling.")]

    def __init__(
        self,
        scale: ScaleFloatType | dict[str, Any] | None = None,
        translate_percent: ScaleFloatType | dict[str, Any] | None = None,
        translate_px: ScaleIntType | dict[str, Any] | None = None,
        rotate: ScaleFloatType | None = None,
        shear: ScaleFloatType | dict[str, Any] | None = None,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        cval: ColorType = 0,
        cval_mask: ColorType = 0,
        mode: int = cv2.BORDER_CONSTANT,
        fit_output: bool = False,
        keep_ratio: bool = False,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        balanced_scale: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        params = [scale, translate_percent, translate_px, rotate, shear]
        if all(p is None for p in params):
            scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)}
            translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
            rotate = (-15, 15)
            shear = {"x": (-10, 10), "y": (-10, 10)}
        else:
            scale = scale if scale is not None else 1.0
            rotate = rotate if rotate is not None else 0.0
            shear = shear if shear is not None else 0.0

        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.scale = self._handle_dict_arg(scale, "scale")
        self.translate_percent, self.translate_px = self._handle_translate_arg(translate_px, translate_percent)
        self.rotate = to_tuple(rotate, rotate)
        self.fit_output = fit_output
        self.shear = self._handle_dict_arg(shear, "shear")
        self.keep_ratio = keep_ratio
        self.rotate_method = rotate_method
        self.balanced_scale = balanced_scale

        if self.keep_ratio and self.scale["x"] != self.scale["y"]:
            raise ValueError(f"When keep_ratio is True, the x and y scale range should be identical. got {self.scale}")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "interpolation",
            "mask_interpolation",
            "cval",
            "mode",
            "scale",
            "translate_percent",
            "translate_px",
            "rotate",
            "fit_output",
            "shear",
            "cval_mask",
            "keep_ratio",
            "rotate_method",
            "balanced_scale",
        )

    @staticmethod
    def _handle_dict_arg(
        val: float | tuple[float, float] | dict[str, Any],
        name: str,
        default: float = 1.0,
    ) -> dict[str, Any]:
        if isinstance(val, dict):
            if "x" not in val and "y" not in val:
                raise ValueError(
                    f'Expected {name} dictionary to contain at least key "x" or key "y". Found neither of them.',
                )
            x = val.get("x", default)
            y = val.get("y", default)
            return {"x": to_tuple(x, x), "y": to_tuple(y, y)}
        return {"x": to_tuple(val, val), "y": to_tuple(val, val)}

    @classmethod
    def _handle_translate_arg(
        cls,
        translate_px: ScaleFloatType | dict[str, Any] | None,
        translate_percent: ScaleFloatType | dict[str, Any] | None,
    ) -> Any:
        if translate_percent is None and translate_px is None:
            translate_px = 0

        if translate_percent is not None and translate_px is not None:
            msg = "Expected either translate_percent or translate_px to be provided, but both were provided."
            raise ValueError(msg)

        if translate_percent is not None:
            # translate by percent
            return cls._handle_dict_arg(translate_percent, "translate_percent", default=0.0), translate_px

        if translate_px is None:
            msg = "translate_px is None."
            raise ValueError(msg)
        # translate by pixels
        return translate_percent, cls._handle_dict_arg(translate_px, "translate_px")

    def apply(
        self,
        img: np.ndarray,
        matrix: skimage.transform.ProjectiveTransform,
        output_shape: SizeType,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.warp_affine(
            img,
            matrix,
            interpolation=self.interpolation,
            cval=self.cval,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        matrix: skimage.transform.ProjectiveTransform,
        output_shape: SizeType,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.warp_affine(
            mask,
            matrix,
            interpolation=self.mask_interpolation,
            cval=self.cval_mask,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        bbox_matrix: skimage.transform.ProjectiveTransform,
        rows: int,
        cols: int,
        output_shape: SizeType,
        **params: Any,
    ) -> BoxInternalType:
        return fgeometric.bbox_affine(bbox, bbox_matrix, self.rotate_method, rows, cols, output_shape)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        matrix: skimage.transform.ProjectiveTransform,
        scale: dict[str, Any],
        **params: Any,
    ) -> KeypointInternalType:
        if scale is None:
            msg = "Expected scale to be provided, but got None."
            raise ValueError(msg)
        if matrix is None:
            msg = "Expected matrix to be provided, but got None."
            raise ValueError(msg)

        return fgeometric.keypoint_affine(keypoint, matrix=matrix, scale=scale)

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    @staticmethod
    def get_scale(scale: dict[str, tuple[float, float]], keep_ratio: bool, balanced_scale: bool) -> dict[str, float]:
        result_scale = {}
        if balanced_scale:
            for key, value in scale.items():
                lower_interval = (value[0], 1.0) if value[0] < 1 else None
                upper_interval = (1.0, value[1]) if value[1] > 1 else None

                if lower_interval is not None and upper_interval is not None:
                    selected_interval = random.choice([lower_interval, upper_interval])
                elif lower_interval is not None:
                    selected_interval = lower_interval
                elif upper_interval is not None:
                    selected_interval = upper_interval
                else:
                    raise ValueError(f"Both lower_interval and upper_interval are None for key: {key}")

                result_scale[key] = random.uniform(*selected_interval)
        else:
            result_scale = {key: random.uniform(*value) for key, value in scale.items()}

        if keep_ratio:
            result_scale["y"] = result_scale["x"]

        return result_scale

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        height, width = params["image"].shape[:2]

        translate: dict[str, int | float]
        if self.translate_px is not None:
            translate = {key: random.randint(*value) for key, value in self.translate_px.items()}
        elif self.translate_percent is not None:
            translate = {key: random.uniform(*value) for key, value in self.translate_percent.items()}
            translate["x"] = translate["x"] * width
            translate["y"] = translate["y"] * height
        else:
            translate = {"x": 0, "y": 0}

        shear = {key: -random.uniform(*value) for key, value in self.shear.items()}

        scale = self.get_scale(self.scale, self.keep_ratio, self.balanced_scale)
        rotate = -random.uniform(*self.rotate)

        shift_x, shift_y = center(width, height)
        shift_x_bbox, shift_y_bbox = center_bbox(width, height)

        # Image transformation matrix
        matrix_to_topleft = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_shear_y_rot = skimage.transform.AffineTransform(rotation=-np.pi / 2)
        matrix_shear_y = skimage.transform.AffineTransform(shear=np.deg2rad(shear["y"]))
        matrix_shear_y_rot_inv = skimage.transform.AffineTransform(rotation=np.pi / 2)
        matrix_transforms = skimage.transform.AffineTransform(
            scale=(scale["x"], scale["y"]),
            translation=(translate["x"], translate["y"]),
            rotation=np.deg2rad(rotate),
            shear=np.deg2rad(shear["x"]),
        )
        matrix_to_center = skimage.transform.SimilarityTransform(translation=[shift_x, shift_y])
        matrix = (
            matrix_to_topleft
            + matrix_shear_y_rot
            + matrix_shear_y
            + matrix_shear_y_rot_inv
            + matrix_transforms
            + matrix_to_center
        )

        # Bounding box transformation matrix
        matrix_to_topleft_bbox = skimage.transform.SimilarityTransform(translation=[-shift_x_bbox, -shift_y_bbox])
        matrix_to_center_bbox = skimage.transform.SimilarityTransform(translation=[shift_x_bbox, shift_y_bbox])
        bbox_matrix = (
            matrix_to_topleft_bbox
            + matrix_shear_y_rot
            + matrix_shear_y
            + matrix_shear_y_rot_inv
            + matrix_transforms
            + matrix_to_center_bbox
        )

        if self.fit_output:
            matrix, output_shape = self._compute_affine_warp_output_shape(matrix, params["image"].shape)
        else:
            output_shape = params["image"].shape

        return {
            "rotate": rotate,
            "scale": scale,
            "matrix": matrix,
            "bbox_matrix": bbox_matrix,
            "output_shape": output_shape,
        }

    @staticmethod
    def _compute_affine_warp_output_shape(
        matrix: skimage.transform.ProjectiveTransform,
        input_shape: SizeType,
    ) -> tuple[skimage.transform.ProjectiveTransform, SizeType]:
        height, width = input_shape[:2]

        if height == 0 or width == 0:
            return matrix, input_shape

        # determine shape of output image
        corners = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
        corners = matrix(corners)

        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()

        out_height = maxr - minr + 1
        out_width = maxc - minc + 1

        if len(input_shape) == NUM_MULTI_CHANNEL_DIMENSIONS:
            output_shape = np.ceil((out_height, out_width, input_shape[2]))
        else:
            output_shape = np.ceil((out_height, out_width))

        output_shape_tuple = tuple(int(v) for v in output_shape.tolist())
        # fit output image in new shape
        translation = -minc, -minr
        matrix_to_fit = skimage.transform.SimilarityTransform(translation=translation)
        matrix += matrix_to_fit
        return matrix, output_shape_tupleclass PadIfNeeded(DualTransform):
    """Pads the sides of an image if the image dimensions are less than the specified minimum dimensions.
    If the `pad_height_divisor` or `pad_width_divisor` is specified, the function additionally ensures
    that the image dimensions are divisible by these values.

    Args:
        min_height (int): Minimum desired height of the image. Ensures image height is at least this value.
        min_width (int): Minimum desired width of the image. Ensures image width is at least this value.
        pad_height_divisor (int, optional): If set, pads the image height to make it divisible by this value.
        pad_width_divisor (int, optional): If set, pads the image width to make it divisible by this value.
        position (Union[str, PositionType]): Position where the image is to be placed after padding.
            Can be one of 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', or 'random'.
            Default is 'center'.
        border_mode (int): Specifies the border mode to use if padding is required.
            The default is `cv2.BORDER_REFLECT_101`. If `value` is provided and `border_mode` is set to a mode
            that does not use a constant value, it should be manually set to `cv2.BORDER_CONSTANT`.
        value (Union[int, float, list[int], list[float]], optional): Value to fill the border pixels if
            the border mode is `cv2.BORDER_CONSTANT`. Default is None.
        mask_value (Union[int, float, list[int], list[float]], optional): Similar to `value` but used for padding masks.
            Default is None.
        p (float): Probability of applying the transform. Default is 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    class PositionType(Enum):
        """Enumerates the types of positions for placing an object within a container.

        This Enum class is utilized to define specific anchor positions that an object can
        assume relative to a container. It's particularly useful in image processing, UI layout,
        and graphic design to specify the alignment and positioning of elements.

        Attributes:
            CENTER (str): Specifies that the object should be placed at the center.
            TOP_LEFT (str): Specifies that the object should be placed at the top-left corner.
            TOP_RIGHT (str): Specifies that the object should be placed at the top-right corner.
            BOTTOM_LEFT (str): Specifies that the object should be placed at the bottom-left corner.
            BOTTOM_RIGHT (str): Specifies that the object should be placed at the bottom-right corner.
            RANDOM (str): Indicates that the object's position should be determined randomly.

        """

        CENTER = "center"
        TOP_LEFT = "top_left"
        TOP_RIGHT = "top_right"
        BOTTOM_LEFT = "bottom_left"
        BOTTOM_RIGHT = "bottom_right"
        RANDOM = "random"

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        min_height: int | None = Field(default=None, ge=1, description="Minimal result image height.")
        min_width: int | None = Field(default=None, ge=1, description="Minimal result image width.")
        pad_height_divisor: int | None = Field(
            default=None,
            ge=1,
            description="Ensures image height is divisible by this value.",
        )
        pad_width_divisor: int | None = Field(
            default=None,
            ge=1,
            description="Ensures image width is divisible by this value.",
        )
        position: str = Field(default="center", description="Position of the padded image.")
        border_mode: BorderModeType = cv2.BORDER_REFLECT_101
        value: ColorType | None = Field(default=None, description="Value for border if BORDER_CONSTANT is used.")
        mask_value: ColorType | None = Field(
            default=None,
            description="Value for mask border if BORDER_CONSTANT is used.",
        )
        p: ProbabilityType = 1.0

        @model_validator(mode="after")
        def validate_divisibility(self) -> Self:
            if (self.min_height is None) == (self.pad_height_divisor is None):
                msg = "Only one of 'min_height' and 'pad_height_divisor' parameters must be set"
                raise ValueError(msg)
            if (self.min_width is None) == (self.pad_width_divisor is None):
                msg = "Only one of 'min_width' and 'pad_width_divisor' parameters must be set"
                raise ValueError(msg)

            if self.value is not None and self.border_mode in {cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT101}:
                self.border_mode = cv2.BORDER_CONSTANT

            if self.border_mode == cv2.BORDER_CONSTANT and self.value is None:
                msg = "If 'border_mode' is set to 'BORDER_CONSTANT', 'value' must be provided."
                raise ValueError(msg)

            return self

    def __init__(
        self,
        min_height: int | None = 1024,
        min_width: int | None = 1024,
        pad_height_divisor: int | None = None,
        pad_width_divisor: int | None = None,
        position: PositionType | str = PositionType.CENTER,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p, always_apply)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = PadIfNeeded.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        params = super().update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_top = int((self.min_height - rows) / 2.0)
                h_pad_bottom = self.min_height - rows - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            pad_rows = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0

            h_pad_top = pad_rows // 2
            h_pad_bottom = pad_rows - h_pad_top

        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_left = int((self.min_width - cols) / 2.0)
                w_pad_right = self.min_width - cols - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            pad_cols = self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0

            w_pad_left = pad_cols // 2
            w_pad_right = pad_cols - w_pad_left

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = self.__update_position_params(
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
        )

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
            },
        )
        return params

    def apply(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.pad_with_params(
            mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        rows: int,
        cols: int,
        **params: Any,
    ) -> BoxInternalType:
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return cast(BoxInternalType, normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right))

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> KeypointInternalType:
        x, y, angle, scale = keypoint[:4]
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "position",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self,
        h_top: int,
        h_bottom: int,
        w_left: int,
        w_right: int,
    ) -> tuple[int, int, int, int]:
        if self.position == PadIfNeeded.PositionType.TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            h_top = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            h_top = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            h_bottom = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            h_bottom = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.RANDOM:
            h_pad = h_top + h_bottom
            w_pad = w_left + w_right
            h_top = random.randint(0, h_pad)
            h_bottom = h_pad - h_top
            w_left = random.randint(0, w_pad)
            w_right = w_pad - w_left

        return h_top, h_bottom, w_left, w_right