    def parse(self, serialized_example: tf_tensor) -> Tuple[tf_tensor]:
        """
        Parse one example
        :param serialized_example: single TFRecord file
        :return: Tuple of multiple Input Images tensors followed by their corresponding labels
        """
        features = tf.parse_single_example(serialized_example,
                                           features={'image/class/label': tf.FixedLenFeature([], tf.int64),
                                                     'image/encoded': tf.FixedLenFeature([], tf.string)})
        image_data = features['image/encoded']
        label = tf.cast(features['image/class/label'], tf.int32) - 1
        labels = tf.one_hot(indices=label, depth=image_net_config.dataset['images_classes'])

        # Decode the jpeg
        with tf.name_scope('prep_image', values=[image_data], default_name=None):
            # decode and reshape to default self._image_size x self._image_size
            # pylint: disable=no-member
            image = tf.image.decode_jpeg(image_data, channels=image_net_config.dataset['image_channels'])
            image = self._preprocess_image(image, self._image_size, self._image_size, is_training=self._is_training)
            if self._format_bgr:
                image = tf.reverse(image, axis=[-1])

        return (image,) + (labels,)    def parse(self, serialized_example: tf_tensor) -> Tuple[tf_tensor]:
        """
        Parse one example
        :param serialized_example: single TFRecord file
        :return: Tuple of multiple Input Images tensors followed by their corresponding labels
        """
        features = tf.parse_single_example(serialized_example,
                                           features={'image/class/label': tf.FixedLenFeature([], tf.int64),
                                                     'image/encoded': tf.FixedLenFeature([], tf.string)})
        image_data = features['image/encoded']
        label = tf.cast(features['image/class/label'], tf.int32) - 1
        labels = tf.one_hot(indices=label, depth=image_net_config.dataset['images_classes'])

        # Decode the jpeg
        with tf.name_scope('prep_image', values=[image_data], default_name=None):
            # decode and reshape to default self._image_size x self._image_size
            # pylint: disable=no-member
            image = tf.image.decode_jpeg(image_data, channels=image_net_config.dataset['image_channels'])
            image = self._preprocess_image(image, self._image_size, self._image_size, is_training=self._is_training)
            if self._format_bgr:
                image = tf.reverse(image, axis=[-1])

        return (image,) + (labels,)