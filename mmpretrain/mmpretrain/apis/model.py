    def get(cls, model_name):
        """Get the model's metainfo by the model name.

        Args:
            model_name (str): The name of model.

        Returns:
            modelindex.models.Model: The metainfo of the specified model.
        """
        cls._register_mmpretrain_models()
        # lazy load config
        metainfo = copy.deepcopy(cls._models_dict.get(model_name.lower()))
        if metainfo is None:
            raise ValueError(
                f'Failed to find model "{model_name}". please use '
                '`mmpretrain.list_models` to get all available names.')
        if isinstance(metainfo.config, str):
            metainfo.config = Config.fromfile(metainfo.config)
        return metainfo