    def get_config(self):
        config = {
            "trainable_kernel": self.trainable_kernel,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))