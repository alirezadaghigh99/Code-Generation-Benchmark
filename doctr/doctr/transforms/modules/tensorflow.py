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