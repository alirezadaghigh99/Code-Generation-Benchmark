class LossWrapper(tf.keras.losses.Loss):
    """ A wrapper class for multiple keras losses to enable multiple masked weighted loss
    functions on a single output.

    Notes
    -----
    Whilst Keras does allow for applying multiple weighted loss functions, it does not allow
    for an easy mechanism to add additional data (in our case masks) that are batch specific
    but are not fed in to the model.

    This wrapper receives this additional mask data for the batch stacked onto the end of the
    color channels of the received :attr:`y_true` batch of images. These masks are then split
    off the batch of images and applied to both the :attr:`y_true` and :attr:`y_pred` tensors
    prior to feeding into the loss functions.

    For example, for an image of shape (4, 128, 128, 3) 3 additional masks may be stacked onto
    the end of y_true, meaning we receive an input of shape (4, 128, 128, 6). This wrapper then
    splits off (4, 128, 128, 3:6) from the end of the tensor, leaving the original y_true of
    shape (4, 128, 128, 3) ready for masking and feeding through the loss functions.
    """
    def __init__(self) -> None:
        logger.debug("Initializing: %s", self.__class__.__name__)
        super().__init__(name="LossWrapper")
        self._loss_functions: list[compile_utils.LossesContainer] = []
        self._loss_weights: list[float] = []
        self._mask_channels: list[int] = []
        logger.debug("Initialized: %s", self.__class__.__name__)

    def add_loss(self,
                 function: Callable,
                 weight: float = 1.0,
                 mask_channel: int = -1) -> None:
        """ Add the given loss function with the given weight to the loss function chain.

        Parameters
        ----------
        function: :class:`tf.keras.losses.Loss`
            The loss function to add to the loss chain
        weight: float, optional
            The weighting to apply to the loss function. Default: `1.0`
        mask_channel: int, optional
            The channel in the `y_true` image that the mask exists in. Set to `-1` if there is no
            mask for the given loss function. Default: `-1`
        """
        logger.debug("Adding loss: (function: %s, weight: %s, mask_channel: %s)",
                     function, weight, mask_channel)
        # Loss must be compiled inside LossContainer for keras to handle distibuted strategies
        self._loss_functions.append(compile_utils.LossesContainer(function))
        self._loss_weights.append(weight)
        self._mask_channels.append(mask_channel)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Call the sub loss functions for the loss wrapper.

        Loss is returned as the weighted sum of the chosen losses.

        If masks are being applied to the loss function inputs, then they should be included as
        additional channels at the end of :attr:`y_true`, so that they can be split off and
        applied to the actual inputs to the selected loss function(s).

        Parameters
        ----------
        y_true: :class:`tensorflow.Tensor`
            The ground truth batch of images, with any required masks stacked on the end
        y_pred: :class:`tensorflow.Tensor`
            The batch of model predictions

        Returns
        -------
        :class:`tensorflow.Tensor`
            The final weighted loss
        """
        loss = 0.0
        for func, weight, mask_channel in zip(self._loss_functions,
                                              self._loss_weights,
                                              self._mask_channels):
            logger.debug("Processing loss function: (func: %s, weight: %s, mask_channel: %s)",
                         func, weight, mask_channel)
            n_true, n_pred = self._apply_mask(y_true, y_pred, mask_channel)
            loss += (func(n_true, n_pred) * weight)
        return loss

    @classmethod
    def _apply_mask(cls,
                    y_true: tf.Tensor,
                    y_pred: tf.Tensor,
                    mask_channel: int,
                    mask_prop: float = 1.0) -> tuple[tf.Tensor, tf.Tensor]:
        """ Apply the mask to the input y_true and y_pred. If a mask is not required then
        return the unmasked inputs.

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value
        y_pred: tensor or variable
            The predicted value
        mask_channel: int
            The channel within y_true that the required mask resides in
        mask_prop: float, optional
            The amount of mask propagation. Default: `1.0`

        Returns
        -------
        tf.Tensor
            The ground truth batch of images, with the required mask applied
        tf.Tensor
            The predicted batch of images with the required mask applied
        """
        if mask_channel == -1:
            logger.debug("No mask to apply")
            return y_true[..., :3], y_pred[..., :3]

        logger.debug("Applying mask from channel %s", mask_channel)

        mask = K.tile(K.expand_dims(y_true[..., mask_channel], axis=-1), (1, 1, 1, 3))
        mask_as_k_inv_prop = 1 - mask_prop
        mask = (mask * mask_prop) + mask_as_k_inv_prop

        m_true = y_true[..., :3] * mask
        m_pred = y_pred[..., :3] * mask

        return m_true, m_pred

