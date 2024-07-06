def is_prediction_registration_forbidden(
    prediction: Prediction,
    persist_predictions: bool,
    roboflow_image_id: Optional[str],
) -> bool:
    return (
        roboflow_image_id is None
        or persist_predictions is False
        or prediction.get("is_stub", False) is True
        or (len(prediction.get("predictions", [])) == 0 and "top" not in prediction)
    )

def register_datapoint_at_roboflow(
    cache: BaseCache,
    strategy_with_spare_credit: str,
    encoded_image: bytes,
    local_image_id: str,
    prediction: Prediction,
    prediction_type: PredictionType,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    inference_id: Optional[str],
) -> None:
    tags = collect_tags(
        configuration=configuration,
        sampling_strategy=strategy_with_spare_credit,
    )
    roboflow_image_id = safe_register_image_at_roboflow(
        cache=cache,
        strategy_with_spare_credit=strategy_with_spare_credit,
        encoded_image=encoded_image,
        local_image_id=local_image_id,
        configuration=configuration,
        api_key=api_key,
        batch_name=batch_name,
        tags=tags,
        inference_id=inference_id,
    )
    if is_prediction_registration_forbidden(
        prediction=prediction,
        persist_predictions=configuration.persist_predictions,
        roboflow_image_id=roboflow_image_id,
    ):
        return None
    encoded_prediction, prediction_file_type = encode_prediction(
        prediction=prediction, prediction_type=prediction_type
    )
    _ = annotate_image_at_roboflow(
        api_key=api_key,
        dataset_id=configuration.dataset_id,
        local_image_id=local_image_id,
        roboflow_image_id=roboflow_image_id,
        annotation_content=encoded_prediction,
        annotation_file_type=prediction_file_type,
        is_prediction=True,
    )

