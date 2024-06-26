def initialize_classes_based_sampling(
    strategy_config: Dict[str, Any]
) -> SamplingMethod:
    try:
        sample_function = partial(
            sample_based_on_classes,
            selected_class_names=set(strategy_config["selected_class_names"]),
            probability=strategy_config["probability"],
        )
        return SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `classes_based_sampling` missing key detected: {error}."
        ) from error