def execute_datapoint_registration(
    cache: BaseCache,
    matching_strategies: List[str],
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    inference_id: Optional[str] = None,
) -> None:
    local_image_id = str(uuid4())
    encoded_image, scaling_factor = prepare_image_to_registration(
        image=image,
        desired_size=configuration.max_image_size,
        jpeg_compression_level=configuration.jpeg_compression_level,
    )
    prediction = adjust_prediction_to_client_scaling_factor(
        prediction=prediction,
        scaling_factor=scaling_factor,
        prediction_type=prediction_type,
    )
    matching_strategies_limits = OrderedDict(
        (strategy_name, configuration.strategies_limits[strategy_name])
        for strategy_name in matching_strategies
    )
    strategy_with_spare_credit = use_credit_of_matching_strategy(
        cache=cache,
        workspace=configuration.workspace_id,
        project=configuration.dataset_id,
        matching_strategies_limits=matching_strategies_limits,
    )
    if strategy_with_spare_credit is None:
        logger.debug(f"Limit on Active Learning strategy reached.")
        return None
    register_datapoint_at_roboflow(
        cache=cache,
        strategy_with_spare_credit=strategy_with_spare_credit,
        encoded_image=encoded_image,
        local_image_id=local_image_id,
        prediction=prediction,
        prediction_type=prediction_type,
        configuration=configuration,
        api_key=api_key,
        batch_name=batch_name,
        inference_id=inference_id,
    )