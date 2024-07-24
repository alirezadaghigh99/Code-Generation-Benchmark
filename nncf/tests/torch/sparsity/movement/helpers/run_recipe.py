class LinearRunRecipe(BaseMockRunRecipe):
    model_family = "linear"
    supports_structured_masking = False
    default_model_config = PretrainedConfig(num_labels=2, input_size=4, bias=True)
    default_algo_config = MovementAlgoConfig(
        MovementSchedulerParams(
            warmup_start_epoch=1,
            warmup_end_epoch=3,
            importance_regularization_factor=0.1,
            enable_structured_masking=False,
            init_importance_threshold=-1.0,
            steps_per_epoch=4,
        )
    )

    def _create_model(self) -> torch.nn.Module:
        model_config = self.model_config
        return LinearForClassification(
            input_size=model_config.input_size, bias=model_config.bias, num_labels=model_config.num_labels
        )

    @property
    def model_input_info(self) -> FillerInputInfo:
        return FillerInputInfo([FillerInputElement(shape=[1, self.model_config.input_size], keyword="tensor")])

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        return []

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(compressed_model: NNCFNetwork) -> List[DictInTransformerBlockOrder]:
        return []

