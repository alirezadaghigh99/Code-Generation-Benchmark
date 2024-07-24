class CrossEntropy(tf.keras.losses.Loss):
    """
    Define weighted cross-entropy.

    The formulation is:
        loss = − w_fg * y_true log(y_pred) - w_bg * (1−y_true) log(1−y_pred)
    """

    def __init__(
        self,
        binary: bool = False,
        background_weight: float = 0.0,
        smooth: float = EPS,
        name: str = "CrossEntropy",
        **kwargs,
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1
        :param background_weight: weight for background, where y == 0.
        :param smooth: smooth constant for log.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        if background_weight < 0 or background_weight > 1:
            raise ValueError(
                "The background weight for Cross Entropy must be "
                f"within [0, 1], got {background_weight}."
            )
        self.binary = binary
        self.background_weight = background_weight
        self.smooth = smooth
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        if self.binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        # (batch, ...) -> (batch, d)
        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)

        loss_fg = -tf.reduce_mean(y_true * tf.math.log(y_pred + self.smooth), axis=1)
        if self.background_weight > 0:
            loss_bg = -tf.reduce_mean(
                (1 - y_true) * tf.math.log(1 - y_pred + self.smooth), axis=1
            )
            return (
                1 - self.background_weight
            ) * loss_fg + self.background_weight * loss_bg
        else:
            return loss_fg

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            binary=self.binary,
            background_weight=self.background_weight,
            smooth=self.smooth,
        )
        return config

