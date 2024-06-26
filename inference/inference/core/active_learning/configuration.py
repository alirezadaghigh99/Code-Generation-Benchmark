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