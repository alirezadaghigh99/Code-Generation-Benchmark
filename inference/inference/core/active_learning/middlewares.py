class ActiveLearningMiddleware:
    @classmethod
    def init(
        cls,
        api_key: str,
        target_dataset: str,
        model_id: str,
        cache: BaseCache,
    ) -> "ActiveLearningMiddleware":
        configuration = prepare_active_learning_configuration(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            cache=cache,
        )
        return cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
        )

    @classmethod
    def init_from_config(
        cls,
        api_key: str,
        target_dataset: str,
        model_id: str,
        cache: BaseCache,
        config: Optional[dict],
    ) -> "ActiveLearningMiddleware":
        configuration = prepare_active_learning_configuration_inplace(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            active_learning_configuration=config,
        )
        return cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
        )

    def __init__(
        self,
        api_key: str,
        configuration: Optional[ActiveLearningConfiguration],
        cache: BaseCache,
    ):
        self._api_key = api_key
        self._configuration = configuration
        self._cache = cache

    def register_batch(
        self,
        inference_inputs: List[Any],
        predictions: List[Prediction],
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
        inference_id=None,
    ) -> None:
        for inference_input, prediction in zip(inference_inputs, predictions):
            self.register(
                inference_input=inference_input,
                prediction=prediction,
                prediction_type=prediction_type,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                inference_id=inference_id,
            )

    def register(
        self,
        inference_input: Any,
        prediction: dict,
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
        inference_id=None,
    ) -> None:
        self._execute_registration(
            inference_input=inference_input,
            prediction=prediction,
            prediction_type=prediction_type,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            inference_id=inference_id,
        )

    def _execute_registration(
        self,
        inference_input: Any,
        prediction: dict,
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
        inference_id=None,
    ) -> None:
        if self._configuration is None:
            return None
        image, is_bgr = load_image(
            value=inference_input,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
        )
        if not is_bgr:
            image = image[:, :, ::-1]
        matching_strategies = execute_sampling(
            image=image,
            prediction=prediction,
            prediction_type=prediction_type,
            sampling_methods=self._configuration.sampling_methods,
        )
        if len(matching_strategies) == 0:
            return None
        batch_name = generate_batch_name(configuration=self._configuration)
        if not image_can_be_submitted_to_batch(
            batch_name=batch_name,
            workspace_id=self._configuration.workspace_id,
            dataset_id=self._configuration.dataset_id,
            max_batch_images=self._configuration.max_batch_images,
            api_key=self._api_key,
        ):
            logger.debug(f"Limit on Active Learning batch size reached.")
            return None
        execute_datapoint_registration(
            cache=self._cache,
            matching_strategies=matching_strategies,
            image=image,
            prediction=prediction,
            prediction_type=prediction_type,
            configuration=self._configuration,
            api_key=self._api_key,
            batch_name=batch_name,
            inference_id=inference_id,
        )

