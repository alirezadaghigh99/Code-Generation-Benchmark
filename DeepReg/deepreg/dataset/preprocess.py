class RandomTransformation3D(tf.keras.layers.Layer):
    """
    An interface for different types of transformation.
    """

    def __init__(
        self,
        moving_image_size: Tuple[int, ...],
        fixed_image_size: Tuple[int, ...],
        batch_size: int,
        name: str = "RandomTransformation3D",
        trainable: bool = False,
    ):
        """
        Abstract class for image transformation.

        :param moving_image_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
        :param batch_size: total number of samples consumed per step, over all devices.
        :param name: name of layer
        :param trainable: if this layer is trainable
        """
        super().__init__(trainable=trainable, name=name)
        self.moving_image_size = moving_image_size
        self.fixed_image_size = fixed_image_size
        self.batch_size = batch_size
        self.moving_grid_ref = get_reference_grid(grid_size=moving_image_size)
        self.fixed_grid_ref = get_reference_grid(grid_size=fixed_image_size)

    @abstractmethod
    def gen_transform_params(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generates transformation parameters for moving and fixed image.

        :return: two tensors
        """

    @staticmethod
    @abstractmethod
    def transform(
        image: tf.Tensor, grid_ref: tf.Tensor, params: tf.Tensor
    ) -> tf.Tensor:
        """
        Transforms the reference grid and then resample the image.

        :param image: shape = (batch, dim1, dim2, dim3)
        :param grid_ref: shape = (dim1, dim2, dim3, 3)
        :param params: parameters for transformation
        :return: shape = (batch, dim1, dim2, dim3)
        """

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs) -> Dict[str, tf.Tensor]:
        """
        Creates random params for the input images and their labels,
        and params them based on the resampled reference grids.
        :param inputs: a dict having multiple tensors
            if labeled:
                moving_image, shape = (batch, m_dim1, m_dim2, m_dim3)
                fixed_image, shape = (batch, f_dim1, f_dim2, f_dim3)
                moving_label, shape = (batch, m_dim1, m_dim2, m_dim3)
                fixed_label, shape = (batch, f_dim1, f_dim2, f_dim3)
                indices, shape = (batch, num_indices)
            else, unlabeled:
                moving_image, shape = (batch, m_dim1, m_dim2, m_dim3)
                fixed_image, shape = (batch, f_dim1, f_dim2, f_dim3)
                indices, shape = (batch, num_indices)
        :param kwargs: other arguments
        :return: dictionary with the same structure as inputs
        """

        moving_image = inputs["moving_image"]
        fixed_image = inputs["fixed_image"]
        indices = inputs["indices"]

        moving_params, fixed_params = self.gen_transform_params()

        moving_image = self.transform(moving_image, self.moving_grid_ref, moving_params)
        fixed_image = self.transform(fixed_image, self.fixed_grid_ref, fixed_params)

        if "moving_label" not in inputs:  # unlabeled
            return dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            )
        moving_label = inputs["moving_label"]
        fixed_label = inputs["fixed_label"]

        moving_label = self.transform(moving_label, self.moving_grid_ref, moving_params)
        fixed_label = self.transform(fixed_label, self.fixed_grid_ref, fixed_params)

        return dict(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            fixed_label=fixed_label,
            indices=indices,
        )

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["moving_image_size"] = self.moving_image_size
        config["fixed_image_size"] = self.fixed_image_size
        config["batch_size"] = self.batch_size
        return config

