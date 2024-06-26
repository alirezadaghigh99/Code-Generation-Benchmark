def initialize_random_sampling(strategy_config: Dict[str, Any]) -> SamplingMethod:
    try:
        sample_function = partial(
            sample_randomly,
            traffic_percentage=strategy_config["traffic_percentage"],
        )
        return SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `random_sampling` missing key detected: {error}."
        ) from error