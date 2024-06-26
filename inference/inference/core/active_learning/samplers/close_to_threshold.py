def initialize_close_to_threshold_sampling(
    strategy_config: Dict[str, Any]
) -> SamplingMethod:
    try:
        selected_class_names = strategy_config.get("selected_class_names")
        if selected_class_names is not None:
            selected_class_names = set(selected_class_names)
        sample_function = partial(
            sample_close_to_threshold,
            selected_class_names=selected_class_names,
            threshold=strategy_config["threshold"],
            epsilon=strategy_config["epsilon"],
            only_top_classes=strategy_config.get("only_top_classes", True),
            minimum_objects_close_to_threshold=strategy_config.get(
                "minimum_objects_close_to_threshold",
                1,
            ),
            probability=strategy_config["probability"],
        )
        return SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `close_to_threshold_sampling` missing key detected: {error}."
        ) from error