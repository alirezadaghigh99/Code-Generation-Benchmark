def sample_based_on_detections_number(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    more_than: Optional[int],
    less_than: Optional[int],
    selected_class_names: Optional[Set[str]],
    probability: float,
) -> bool:
    if is_prediction_a_stub(prediction=prediction):
        return False
    if prediction_type not in ELIGIBLE_PREDICTION_TYPES:
        return False
    detections_close_to_threshold = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names=selected_class_names,
        threshold=0.5,
        epsilon=1.0,
    )
    if is_in_range(
        value=detections_close_to_threshold, less_than=less_than, more_than=more_than
    ):
        return random.random() < probability
    return False