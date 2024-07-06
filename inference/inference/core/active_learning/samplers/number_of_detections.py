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

def is_in_range(
    value: int,
    more_than: Optional[int],
    less_than: Optional[int],
) -> bool:
    # calculates value > more_than and value < less_than, with optional borders of range
    less_than_satisfied, more_than_satisfied = less_than is None, more_than is None
    if less_than is not None and value < less_than:
        less_than_satisfied = True
    if more_than is not None and value > more_than:
        more_than_satisfied = True
    return less_than_satisfied and more_than_satisfied

