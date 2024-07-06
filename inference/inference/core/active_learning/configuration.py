def initialize_sampling_methods(
    sampling_strategies_configs: List[Dict[str, Any]]
) -> List[SamplingMethod]:
    result = []
    for sampling_strategy_config in sampling_strategies_configs:
        sampling_type = sampling_strategy_config["type"]
        if sampling_type not in TYPE2SAMPLING_INITIALIZERS:
            logger.warn(
                f"Could not identify sampling method `{sampling_type}` - skipping initialisation."
            )
            continue
        initializer = TYPE2SAMPLING_INITIALIZERS[sampling_type]
        result.append(initializer(sampling_strategy_config))
    names = set(m.name for m in result)
    if len(names) != len(result):
        raise ActiveLearningConfigurationError(
            "Detected duplication of Active Learning strategies names."
        )
    return result

def predictions_incompatible_with_dataset(
    model_type: str,
    dataset_type: str,
) -> bool:
    """
    The incompatibility occurs when we mix classification with detection - as detection-based
    predictions are partially compatible (for instance - for key-points detection we may register bboxes
    from object detection and manually provide key-points annotations)
    """
    model_is_classifier = CLASSIFICATION_TASK in model_type
    dataset_is_of_type_classification = CLASSIFICATION_TASK in dataset_type
    return model_is_classifier != dataset_is_of_type_classification

