class RegistrationModel(tf.keras.Model):
    """Interface for registration model."""

    def __init__(
        self,
        moving_image_size: Tuple,
        fixed_image_size: Tuple,
        index_size: int,
        labeled: bool,
        batch_size: int,
        config: dict,
        name: str = "RegistrationModel",
    ):
        """
        Init.

        :param moving_image_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
        :param index_size: number of indices for identify each sample
        :param labeled: if the data is labeled
        :param batch_size: total number of samples consumed per step, over all devices.
            When using multiple devices, TensorFlow automatically split the tensors.
            Therefore, input shapes should be defined over batch_size.
        :param config: config for method, backbone, and loss.
        :param name: name of the model
        """
        super().__init__(name=name)
        self.moving_image_size = moving_image_size
        self.fixed_image_size = fixed_image_size
        self.index_size = index_size
        self.labeled = labeled
        self.config = config
        self.batch_size = batch_size

        self._inputs = None  # save inputs of self._model as dict
        self._outputs = None  # save outputs of self._model as dict

        self.grid_ref = layer_util.get_reference_grid(grid_size=fixed_image_size)[
            None, ...
        ]
        self._model: tf.keras.Model = self.build_model()
        self.build_loss()

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        return dict(
            moving_image_size=self.moving_image_size,
            fixed_image_size=self.fixed_image_size,
            index_size=self.index_size,
            labeled=self.labeled,
            batch_size=self.batch_size,
            config=self.config,
            name=self.name,
        )

    @abstractmethod
    def build_model(self):
        """Build the model to be saved as self._model."""

    def build_inputs(self) -> Dict[str, tf.keras.layers.Input]:
        """
        Build input tensors.

        :return: dict of inputs.
        """
        # (batch, m_dim1, m_dim2, m_dim3)
        moving_image = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_image",
        )
        # (batch, f_dim1, f_dim2, f_dim3)
        fixed_image = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_image",
        )
        # (batch, index_size)
        indices = tf.keras.Input(
            shape=(self.index_size,),
            batch_size=self.batch_size,
            name="indices",
        )

        if not self.labeled:
            return dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            )

        # (batch, m_dim1, m_dim2, m_dim3)
        moving_label = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_label",
        )
        # (batch, m_dim1, m_dim2, m_dim3)
        fixed_label = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_label",
        )
        return dict(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            fixed_label=fixed_label,
            indices=indices,
        )

    def concat_images(
        self,
        moving_image: tf.Tensor,
        fixed_image: tf.Tensor,
        moving_label: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Adjust image shape and concatenate them together.

        :param moving_image: registration source
        :param fixed_image: registration target
        :param moving_label: optional, only used for conditional model.
        :return:
        """
        images = []

        resize_layer = layer.Resize3d(shape=self.fixed_image_size)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_image = tf.expand_dims(moving_image, axis=4)
        moving_image = resize_layer(moving_image)
        images.append(moving_image)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        fixed_image = tf.expand_dims(fixed_image, axis=4)
        images.append(fixed_image)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        if moving_label is not None:
            moving_label = tf.expand_dims(moving_label, axis=4)
            moving_label = resize_layer(moving_label)
            images.append(moving_label)

        # (batch, f_dim1, f_dim2, f_dim3, 2 or 3)
        images = tf.concat(images, axis=4)
        return images

    def _build_loss(self, name: str, inputs_dict: dict):
        """
        Build and add one weighted loss together with the metrics.

        :param name: name of loss, image / label / regularization.
        :param inputs_dict: inputs for loss function
        """

        if name not in self.config["loss"]:
            # loss config is not defined
            logger.warning(
                f"The configuration for loss {name} is not defined. "
                f"Therefore it is not used."
            )
            return

        loss_configs = self.config["loss"][name]
        if not isinstance(loss_configs, list):
            loss_configs = [loss_configs]

        for loss_config in loss_configs:

            if "weight" not in loss_config:
                # default loss weight 1
                logger.warning(
                    f"The weight for loss {name} is not defined."
                    f"Default weight = 1.0 is used."
                )
                loss_config["weight"] = 1.0

            # build loss
            weight = loss_config["weight"]

            if weight == 0:
                logger.warning(
                    f"The weight for loss {name} is zero." f"Loss is not used."
                )
                return

            # do not perform reduction over batch axis for supporting multi-device
            # training, model.fit() will average over global batch size automatically
            loss_layer: tf.keras.layers.Layer = REGISTRY.build_loss(
                config=dict_without(d=loss_config, key="weight"),
                default_args={"reduction": tf.keras.losses.Reduction.NONE},
            )
            loss_value = loss_layer(**inputs_dict)
            weighted_loss = loss_value * weight

            # add loss
            self._model.add_loss(weighted_loss)

            # add metric
            self._model.add_metric(
                loss_value, name=f"loss/{name}_{loss_layer.name}", aggregation="mean"
            )
            self._model.add_metric(
                weighted_loss,
                name=f"loss/{name}_{loss_layer.name}_weighted",
                aggregation="mean",
            )

    @abstractmethod
    def build_loss(self):
        """Build losses according to configs."""

        # input metrics
        fixed_image = self._inputs["fixed_image"]
        moving_image = self._inputs["moving_image"]
        self.log_tensor_stats(tensor=moving_image, name="moving_image")
        self.log_tensor_stats(tensor=fixed_image, name="fixed_image")

        # image loss, conditional model does not have this
        if "pred_fixed_image" in self._outputs:
            pred_fixed_image = self._outputs["pred_fixed_image"]
            self._build_loss(
                name="image",
                inputs_dict=dict(y_true=fixed_image, y_pred=pred_fixed_image),
            )

        if self.labeled:
            # input metrics
            fixed_label = self._inputs["fixed_label"]
            moving_label = self._inputs["moving_label"]
            self.log_tensor_stats(tensor=moving_label, name="moving_label")
            self.log_tensor_stats(tensor=fixed_label, name="fixed_label")

            # label loss
            pred_fixed_label = self._outputs["pred_fixed_label"]
            self._build_loss(
                name="label",
                inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
            )

            # additional label metrics
            tre = compute_centroid_distance(
                y_true=fixed_label, y_pred=pred_fixed_label, grid=self.grid_ref
            )
            self._model.add_metric(tre, name="metric/TRE", aggregation="mean")

    def call(
        self, inputs: Dict[str, tf.Tensor], training=None, mask=None
    ) -> Dict[str, tf.Tensor]:
        """
        Call the self._model.

        :param inputs: a dict of tensors.
        :param training: training or not.
        :param mask: maks for inputs.
        :return:
        """
        return self._model(inputs, training=training, mask=mask)  # pragma: no cover

    @abstractmethod
    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, Dict]:
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """

    def plot_model(self, output_dir: str):
        """
        Save model structure in png.

        :param output_dir: path to the output dir.
        """
        self._model.summary(print_fn=logger.debug)
        try:
            tf.keras.utils.plot_model(
                self._model,
                to_file=os.path.join(output_dir, f"{self.name}.png"),
                dpi=96,
                show_shapes=True,
                show_layer_names=True,
                expand_nested=False,
            )
        except ImportError as err:  # pragma: no cover
            logger.error(
                "Failed to plot model structure. "
                "Please check if graphviz is installed. "
                "Error message is: %s.",
                err,
            )

    def log_tensor_stats(self, tensor: tf.Tensor, name: str):
        """
        Log statistics of a given tensor.

        :param tensor: tensor to monitor.
        :param name: name of the tensor.
        """
        flatten = tf.reshape(tensor, shape=(self.batch_size, -1))
        self._model.add_metric(
            tf.reduce_mean(flatten, axis=1),
            name=f"metric/{name}_mean",
            aggregation="mean",
        )
        self._model.add_metric(
            tf.reduce_min(flatten, axis=1),
            name=f"metric/{name}_min",
            aggregation="min",
        )
        self._model.add_metric(
            tf.reduce_max(flatten, axis=1),
            name=f"metric/{name}_max",
            aggregation="max",
        )

