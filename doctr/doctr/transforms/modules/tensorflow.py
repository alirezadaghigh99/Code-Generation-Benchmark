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