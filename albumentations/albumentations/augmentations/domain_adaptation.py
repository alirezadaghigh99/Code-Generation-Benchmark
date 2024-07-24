class PixelDistributionAdaptation(ImageOnlyTransform):
    """Performs pixel-level domain adaptation by aligning the pixel value distribution of an input image
    with that of a reference image. This process involves fitting a simple statistical transformation
    (such as PCA, StandardScaler, or MinMaxScaler) to both the original and the reference images,
    transforming the original image with the transformation trained on it, and then applying the inverse
    transformation using the transform fitted on the reference image. The result is an adapted image
    that retains the original content while mimicking the pixel value distribution of the reference domain.

    The process can be visualized as two main steps:
    1. Adjusting the original image to a standard distribution space using a selected transform.
    2. Moving the adjusted image into the distribution space of the reference image by applying the inverse
       of the transform fitted on the reference image.

    This technique is especially useful in scenarios where images from different domains (e.g., synthetic
    vs. real images, day vs. night scenes) need to be harmonized for better consistency or performance in
    image processing tasks.

    Args:
        reference_images (Sequence[Any]): A sequence of objects (typically image paths) that will be
            converted into images by `read_fn`. These images serve as references for the domain adaptation.
        blend_ratio (tuple[float, float]): Specifies the minimum and maximum blend ratio for mixing
            the adapted image with the original, enhancing the diversity of the output images.
        read_fn (Callable): A user-defined function for reading and converting the objects in
            `reference_images` into numpy arrays. By default, it assumes these objects are image paths.
        transform_type (str): Specifies the type of statistical transformation to apply. Supported values
            are "pca" for Principal Component Analysis, "standard" for StandardScaler, and "minmax" for
            MinMaxScaler.
        p (float): The probability of applying the transform to any given image. Default is 1.0.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        For more information on the underlying approach, see: https://github.com/arsenyinfo/qudida

    Note:
        The PixelDistributionAdaptation transform is a novel way to perform domain adaptation at the pixel level,
        suitable for adjusting images across different conditions without complex modeling. It is effective
        for preparing images before more advanced processing or analysis.
    """

    class InitSchema(BaseTransformInitSchema):
        reference_images: Sequence[Any]
        blend_ratio: ZeroOneRangeType = (0.25, 1.0)
        read_fn: Callable[[Any], np.ndarray]
        transform_type: Literal["pca", "standard", "minmax"]

    def __init__(
        self,
        reference_images: Sequence[Any],
        blend_ratio: tuple[float, float] = (0.25, 1.0),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        transform_type: Literal["pca", "standard", "minmax"] = "pca",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.blend_ratio = blend_ratio
        self.transform_type = transform_type

    @staticmethod
    def _validate_shape(img: np.ndarray) -> None:
        if is_grayscale_image(img) or is_multispectral_image(img):
            raise ValueError(
                f"Unexpected image shape: expected 3 dimensions, got {len(img.shape)}."
                f"Is it a grayscale or multispectral image? It's not supported for now.",
            )

    def ensure_uint8(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        if img.dtype == np.float32:
            if img.min() < 0 or img.max() > 1:
                message = (
                    "PixelDistributionAdaptation uses uint8 under the hood, so float32 should be converted,"
                    "Can not do it automatically when the image is out of [0..1] range."
                )
                raise TypeError(message)
            return clip(img * 255, np.uint8), True
        return img, False

    def apply(self, img: np.ndarray, reference_image: np.ndarray, blend_ratio: float, **params: Any) -> np.ndarray:
        self._validate_shape(img)
        reference_image, _ = self.ensure_uint8(reference_image)
        img, needs_reconvert = self.ensure_uint8(img)

        adapted = adapt_pixel_distribution(
            img,
            ref=reference_image,
            weight=blend_ratio,
            transform_type=self.transform_type,
        )

        return fmain.to_float(adapted) if needs_reconvert else adapted

    def get_params(self) -> dict[str, Any]:
        return {
            "reference_image": self.read_fn(random.choice(self.reference_images)),
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def get_transform_init_args_names(self) -> tuple[str, str, str, str]:
        return "reference_images", "blend_ratio", "read_fn", "transform_type"

    def to_dict_private(self) -> dict[str, Any]:
        msg = "PixelDistributionAdaptation can not be serialized."
        raise NotImplementedError(msg)

