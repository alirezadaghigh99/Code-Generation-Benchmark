class BSplines3DTransform(tfkl.Layer):
    """
    Layer for BSplines interpolation with precomputed cubic spline kernel_size.
    It assumes a full sized image from which:
    1. it compute the contol points values by down-sampling the initial image
    2. performs the interpolation
    3. crops the image around the valid values.
    """

    def __init__(
        self,
        cp_spacing: Union[Tuple[int, ...], int],
        output_shape: Tuple[int, ...],
        **kwargs,
    ):
        """
        Init.

        :param cp_spacing: int or tuple of three ints specifying the spacing (in pixels)
            in each dimension. When a single int is used,
            the same spacing to all dimensions is used
        :param output_shape: (batch_size, dim0, dim1, dim2, 3) of the high resolution
            deformation fields.
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)

        self._output_shape = output_shape
        if isinstance(cp_spacing, int):
            cp_spacing = (cp_spacing, cp_spacing, cp_spacing)
        self.cp_spacing = cp_spacing

    def build(self, input_shape: tuple):
        """
        :param input_shape: tuple with the input shape
        :return: None
        """

        super().build(input_shape=input_shape)

        b = {
            0: lambda u: np.float64((1 - u) ** 3 / 6),
            1: lambda u: np.float64((3 * (u ** 3) - 6 * (u ** 2) + 4) / 6),
            2: lambda u: np.float64((-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6),
            3: lambda u: np.float64(u ** 3 / 6),
        }

        filters = np.zeros(
            (
                4 * self.cp_spacing[0],
                4 * self.cp_spacing[1],
                4 * self.cp_spacing[2],
                3,
                3,
            ),
            dtype=np.float32,
        )

        u_arange = 1 - np.arange(
            1 / (2 * self.cp_spacing[0]), 1, 1 / self.cp_spacing[0]
        )
        v_arange = 1 - np.arange(
            1 / (2 * self.cp_spacing[1]), 1, 1 / self.cp_spacing[1]
        )
        w_arange = 1 - np.arange(
            1 / (2 * self.cp_spacing[2]), 1, 1 / self.cp_spacing[2]
        )

        filter_idx = [[0, 1, 2, 3] for _ in range(3)]
        filter_coord = list(itertools.product(*filter_idx))

        for f_idx in filter_coord:
            for it_dim in range(3):
                filters[
                    f_idx[0] * self.cp_spacing[0] : (f_idx[0] + 1) * self.cp_spacing[0],
                    f_idx[1] * self.cp_spacing[1] : (f_idx[1] + 1) * self.cp_spacing[1],
                    f_idx[2] * self.cp_spacing[2] : (f_idx[2] + 1) * self.cp_spacing[2],
                    it_dim,
                    it_dim,
                ] = (
                    b[f_idx[0]](u_arange)[:, None, None]
                    * b[f_idx[1]](v_arange)[None, :, None]
                    * b[f_idx[2]](w_arange)[None, None, :]
                )

        self.filter = tf.convert_to_tensor(filters)

    def interpolate(self, field) -> tf.Tensor:
        """
        :param field: tf.Tensor with shape=number_of_control_points_per_dim
        :return: interpolated_field: tf.Tensor
        """

        image_shape = tuple(
            [(a - 1) * b + 4 * b for a, b in zip(field.shape[1:-1], self.cp_spacing)]
        )

        output_shape = (field.shape[0],) + image_shape + (3,)
        return tf.nn.conv3d_transpose(
            field,
            self.filter,
            output_shape=output_shape,
            strides=self.cp_spacing,
            padding="VALID",
        )

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """
        :param inputs: tf.Tensor defining a low resolution free-form deformation field
        :param kwargs: additional arguments.
        :return: interpolated_field: tf.Tensor of shape=self.input_shape
        """
        high_res_field = self.interpolate(inputs)

        index = [int(3 * c) for c in self.cp_spacing]
        return high_res_field[
            :,
            index[0] : index[0] + self._output_shape[0],
            index[1] : index[1] + self._output_shape[1],
            index[2] : index[2] + self._output_shape[2],
        ]

class Resize3d(tfkl.Layer):
    """
    Resize image in two folds.

    - resize dim2 and dim3
    - resize dim1 and dim2
    """

    def __init__(
        self,
        shape: tuple,
        method: str = tf.image.ResizeMethod.BILINEAR,
        name: str = "resize3d",
    ):
        """
        Init, save arguments.

        :param shape: (dim1, dim2, dim3)
        :param method: tf.image.ResizeMethod
        :param name: name of the layer
        """
        super().__init__(name=name)
        assert len(shape) == 3
        self._shape = shape
        self._method = method

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Perform two fold resize.

        :param inputs: shape = (batch, dim1, dim2, dim3, channels)
                                     or (batch, dim1, dim2, dim3)
                                     or (dim1, dim2, dim3)
        :param kwargs: additional arguments
        :return: shape = (batch, out_dim1, out_dim2, out_dim3, channels)
                                or (batch, dim1, dim2, dim3)
                                or (dim1, dim2, dim3)
        """
        # sanity check
        image = inputs
        image_dim = len(image.shape)

        # init
        if image_dim == 5:
            has_channel = True
            has_batch = True
            input_image_shape = image.shape[1:4]
        elif image_dim == 4:
            has_channel = False
            has_batch = True
            input_image_shape = image.shape[1:4]
        elif image_dim == 3:
            has_channel = False
            has_batch = False
            input_image_shape = image.shape[0:3]
        else:
            raise ValueError(
                "Resize3d takes input image of dimension 3 or 4 or 5, "
                "corresponding to (dim1, dim2, dim3) "
                "or (batch, dim1, dim2, dim3) "
                "or (batch, dim1, dim2, dim3, channels), "
                "got image shape{}".format(image.shape)
            )

        # no need of resize
        if input_image_shape == tuple(self._shape):
            return image

        # expand to five dimensions
        if not has_batch:
            image = tf.expand_dims(image, axis=0)
        if not has_channel:
            image = tf.expand_dims(image, axis=-1)
        assert len(image.shape) == 5  # (batch, dim1, dim2, dim3, channels)
        image_shape = tf.shape(image)

        # merge axis 0 and 1
        output = tf.reshape(
            image, (-1, image_shape[2], image_shape[3], image_shape[4])
        )  # (batch * dim1, dim2, dim3, channels)

        # resize dim2 and dim3
        output = tf.image.resize(
            images=output, size=self._shape[1:3], method=self._method
        )  # (batch * dim1, out_dim2, out_dim3, channels)

        # split axis 0 and merge axis 3 and 4
        output = tf.reshape(
            output,
            shape=(-1, image_shape[1], self._shape[1], self._shape[2] * image_shape[4]),
        )  # (batch, dim1, out_dim2, out_dim3 * channels)

        # resize dim1 and dim2
        output = tf.image.resize(
            images=output, size=self._shape[:2], method=self._method
        )  # (batch, out_dim1, out_dim2, out_dim3 * channels)

        # reshape
        output = tf.reshape(
            output, shape=[-1, *self._shape, image_shape[4]]
        )  # (batch, out_dim1, out_dim2, out_dim3, channels)

        # squeeze to original dimension
        if not has_batch:
            output = tf.squeeze(output, axis=0)
        if not has_channel:
            output = tf.squeeze(output, axis=-1)
        return output

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["shape"] = self._shape
        config["method"] = self._method
        return config

class ResizeCPTransform(tfkl.Layer):
    """
    Layer for getting the control points from the output of a image-to-image network.
    It uses an anti-aliasing Gaussian filter before down-sampling.
    """

    def __init__(
        self, control_point_spacing: Union[List[int], Tuple[int, ...], int], **kwargs
    ):
        """
        :param control_point_spacing: list or int
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)

        if isinstance(control_point_spacing, int):
            control_point_spacing = [control_point_spacing] * 3

        self.kernel_sigma = [
            0.44 * cp for cp in control_point_spacing
        ]  # 0.44 = ln(4)/pi
        self.cp_spacing = control_point_spacing
        self.kernel = None
        self._output_shape = None
        self._resize = None

    def build(self, input_shape):
        super().build(input_shape=input_shape)

        self.kernel = layer_util.gaussian_filter_3d(self.kernel_sigma)
        output_shape = tuple(
            tf.cast(tf.math.ceil(v / c) + 3, tf.int32)
            for v, c in zip(input_shape[1:-1], self.cp_spacing)
        )
        self._output_shape = output_shape
        self._resize = Resize3d(output_shape)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        output = tf.nn.conv3d(
            inputs, self.kernel, strides=(1, 1, 1, 1, 1), padding="SAME"
        )
        output = self._resize(inputs=output)  # type: ignore
        return output

class NormBlock(tfkl.Layer):
    """
    A block with layer - norm - activation.
    """

    def __init__(
        self,
        layer_name: str,
        norm_name: str = "batch",
        activation: str = "relu",
        name: str = "norm_block",
        **kwargs,
    ):
        """
        Init.

        :param layer_name: class of the layer to be wrapped.
        :param norm_name: class of the normalization layer.
        :param activation: name of activation.
        :param name: name of the block layer.
        :param kwargs: additional arguments.
        """
        super().__init__()
        self._config = dict(
            layer_name=layer_name,
            norm_name=norm_name,
            activation=activation,
            name=name,
            **kwargs,
        )
        self._layer = LAYER_DICT[layer_name](use_bias=False, **kwargs)
        self._norm = NORM_DICT[norm_name]()
        self._act = tfkl.Activation(activation=activation)

    def call(self, inputs, training=None, **kwargs) -> tf.Tensor:
        """
        Forward.

        :param inputs: inputs for the layer
        :param training: training flag for normalization layers (default: None)
        :param kwargs: additional arguments.
        :return:
        """
        output = self._layer(inputs=inputs)
        output = self._norm(inputs=output, training=training)
        output = self._act(output)
        return output

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(self._config)
        return config

class ResidualBlock(tfkl.Layer):
    """
    A block with skip links and layer - norm - activation.
    """

    def __init__(
        self,
        layer_name: str,
        num_layers: int = 2,
        norm_name: str = "batch",
        activation: str = "relu",
        name: str = "res_block",
        **kwargs,
    ):
        """
        Init.

        :param layer_name: class of the layer to be wrapped.
        :param num_layers: number of layers/blocks.
        :param norm_name: class of the normalization layer.
        :param activation: name of activation.
        :param name: name of the block layer.
        :param kwargs: additional arguments.
        """
        super().__init__()
        self._num_layers = num_layers
        self._config = dict(
            layer_name=layer_name,
            num_layers=num_layers,
            norm_name=norm_name,
            activation=activation,
            name=name,
            **kwargs,
        )
        self._layers = [
            LAYER_DICT[layer_name](use_bias=False, **kwargs) for _ in range(num_layers)
        ]
        self._norms = [NORM_DICT[norm_name]() for _ in range(num_layers)]
        self._acts = [tfkl.Activation(activation=activation) for _ in range(num_layers)]

    def call(self, inputs, training=None, **kwargs) -> tf.Tensor:
        """
        Forward.

        :param inputs: inputs for the layer
        :param training: training flag for normalization layers (default: None)
        :param kwargs: additional arguments.
        :return:
        """

        output = inputs
        for i in range(self._num_layers):
            output = self._layers[i](inputs=output)
            output = self._norms[i](inputs=output, training=training)
            if i == self._num_layers - 1:
                # last block
                output = output + inputs
            output = self._acts[i](output)
        return output

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(self._config)
        return config

