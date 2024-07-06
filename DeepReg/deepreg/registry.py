def build_model(self, config: Dict, default_args: Optional[dict] = None) -> Any:
        """
        Instantiate a registered model class.

        :param config: config having key `name`.
        :param default_args: optionally some default arguments.
        :return: a model instance
        """
        return self.build_from_config(
            category=MODEL_CLASS, config=config, default_args=default_args
        )

