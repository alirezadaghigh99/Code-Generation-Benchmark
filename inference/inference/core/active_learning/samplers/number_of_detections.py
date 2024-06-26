def initialize_detections_number_based_sampling(
    strategy_config: Dict[str, Any]
) -> SamplingMethod:
    try:
        more_than = strategy_config.get("more_than")
        less_than = strategy_config.get("less_than")
        ensure_range_configuration_is_valid(more_than=more_than, less_than=less_than)
        selected_class_names = strategy_config.get("selected_class_names")
        if selected_class_names is not None:
            selected_class_names = set(selected_class_names)
        sample_function = partial(
            sample_based_on_detections_number,
            less_than=less_than,
            more_than=more_than,
            selected_class_names=selected_class_names,
            probability=strategy_config["probability"],
        )
        return SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `detections_number_based_sampling` missing key detected: {error}."
        ) from error