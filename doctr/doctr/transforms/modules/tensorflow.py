class Normalize(NestedObject):
    """Normalize a tensor to a Gaussian distribution for each channel

    >>> import tensorflow as tf
    >>> from doctr.transforms import Normalize
    >>> transfo = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        mean: average value per channel
        std: standard deviation per channel
    """

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = tf.constant(mean)
        self.std = tf.constant(std)

    def extra_repr(self) -> str:
        return f"mean={self.mean.numpy().tolist()}, std={self.std.numpy().tolist()}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        img -= tf.cast(self.mean, dtype=img.dtype)
        img /= tf.cast(self.std, dtype=img.dtype)
        return img

class LambdaTransformation(NestedObject):
    """Normalize a tensor to a Gaussian distribution for each channel

    >>> import tensorflow as tf
    >>> from doctr.transforms import LambdaTransformation
    >>> transfo = LambdaTransformation(lambda x: x/ 255.)
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        fn: the function to be applied to the input tensor
    """

    def __init__(self, fn: Callable[[tf.Tensor], tf.Tensor]) -> None:
        self.fn = fn

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return self.fn(img)

class ToGray(NestedObject):
    """Convert a RGB tensor (batch of images or image) to a 3-channels grayscale tensor

    >>> import tensorflow as tf
    >>> from doctr.transforms import ToGray
    >>> transfo = ToGray()
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))
    """

    def __init__(self, num_output_channels: int = 1):
        self.num_output_channels = num_output_channels

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        img = tf.image.rgb_to_grayscale(img)
        return img if self.num_output_channels == 1 else tf.repeat(img, self.num_output_channels, axis=-1)

class RandomBrightness(NestedObject):
    """Randomly adjust brightness of a tensor (batch of images or image) by adding a delta
    to all pixels

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomBrightness
    >>> transfo = RandomBrightness()
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        max_delta: offset to add to each pixel is randomly picked in [-max_delta, max_delta]
        p: probability to apply transformation
    """

    def __init__(self, max_delta: float = 0.3) -> None:
        self.max_delta = max_delta

    def extra_repr(self) -> str:
        return f"max_delta={self.max_delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_brightness(img, max_delta=self.max_delta)

class RandomContrast(NestedObject):
    """Randomly adjust contrast of a tensor (batch of images or image) by adjusting
    each pixel: (img - mean) * contrast_factor + mean.

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomContrast
    >>> transfo = RandomContrast()
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        delta: multiplicative factor is picked in [1-delta, 1+delta] (reduce contrast if factor<1)
    """

    def __init__(self, delta: float = 0.3) -> None:
        self.delta = delta

    def extra_repr(self) -> str:
        return f"delta={self.delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_contrast(img, lower=1 - self.delta, upper=1 / (1 - self.delta))

class RandomSaturation(NestedObject):
    """Randomly adjust saturation of a tensor (batch of images or image) by converting to HSV and
    increasing saturation by a factor.

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomSaturation
    >>> transfo = RandomSaturation()
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        delta: multiplicative factor is picked in [1-delta, 1+delta] (reduce saturation if factor<1)
    """

    def __init__(self, delta: float = 0.5) -> None:
        self.delta = delta

    def extra_repr(self) -> str:
        return f"delta={self.delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_saturation(img, lower=1 - self.delta, upper=1 + self.delta)

class RandomHue(NestedObject):
    """Randomly adjust hue of a tensor (batch of images or image) by converting to HSV and adding a delta

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomHue
    >>> transfo = RandomHue()
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        max_delta: offset to add to each pixel is randomly picked in [-max_delta, max_delta]
    """

    def __init__(self, max_delta: float = 0.3) -> None:
        self.max_delta = max_delta

    def extra_repr(self) -> str:
        return f"max_delta={self.max_delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_hue(img, max_delta=self.max_delta)

class RandomGamma(NestedObject):
    """randomly performs gamma correction for a tensor (batch of images or image)

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomGamma
    >>> transfo = RandomGamma()
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        min_gamma: non-negative real number, lower bound for gamma param
        max_gamma: non-negative real number, upper bound for gamma
        min_gain: lower bound for constant multiplier
        max_gain: upper bound for constant multiplier
    """

    def __init__(
        self,
        min_gamma: float = 0.5,
        max_gamma: float = 1.5,
        min_gain: float = 0.8,
        max_gain: float = 1.2,
    ) -> None:
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.min_gain = min_gain
        self.max_gain = max_gain

    def extra_repr(self) -> str:
        return f"""gamma_range=({self.min_gamma}, {self.max_gamma}),
                 gain_range=({self.min_gain}, {self.max_gain})"""

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        gain = random.uniform(self.min_gain, self.max_gain)
        return tf.image.adjust_gamma(img, gamma=gamma, gain=gain)

class RandomJpegQuality(NestedObject):
    """Randomly adjust jpeg quality of a 3 dimensional RGB image

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomJpegQuality
    >>> transfo = RandomJpegQuality()
    >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        min_quality: int between [0, 100]
        max_quality: int between [0, 100]
    """

    def __init__(self, min_quality: int = 60, max_quality: int = 100) -> None:
        self.min_quality = min_quality
        self.max_quality = max_quality

    def extra_repr(self) -> str:
        return f"min_quality={self.min_quality}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_jpeg_quality(img, min_jpeg_quality=self.min_quality, max_jpeg_quality=self.max_quality)

class GaussianBlur(NestedObject):
    """Randomly adjust jpeg quality of a 3 dimensional RGB image

    >>> import tensorflow as tf
    >>> from doctr.transforms import GaussianBlur
    >>> transfo = GaussianBlur(3, (.1, 5))
    >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        kernel_shape: size of the blurring kernel
        std: min and max value of the standard deviation
    """

    def __init__(self, kernel_shape: Union[int, Iterable[int]], std: Tuple[float, float]) -> None:
        self.kernel_shape = kernel_shape
        self.std = std

    def extra_repr(self) -> str:
        return f"kernel_shape={self.kernel_shape}, std={self.std}"

    @tf.function
    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(
            _gaussian_filter(
                img[tf.newaxis, ...],
                kernel_size=self.kernel_shape,
                sigma=random.uniform(self.std[0], self.std[1]),
                mode="REFLECT",
            ),
            axis=0,
        )

class Compose(NestedObject):
    """Implements a wrapper that will apply transformations sequentially

    >>> import tensorflow as tf
    >>> from doctr.transforms import Compose, Resize
    >>> transfos = Compose([Resize((32, 32))])
    >>> out = transfos(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
    ----
        transforms: list of transformation modules
    """

    _children_names: List[str] = ["transforms"]

    def __init__(self, transforms: List[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)

        return x

