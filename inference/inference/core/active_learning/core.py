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

def prepare_image_to_registration(
    image: np.ndarray,
    desired_size: Optional[ImageDimensions],
    jpeg_compression_level: int,
) -> Tuple[bytes, float]:
    scaling_factor = 1.0
    if desired_size is not None:
        height_before_scale = image.shape[0]
        image = downscale_image_keeping_aspect_ratio(
            image=image,
            desired_size=desired_size.to_wh(),
        )
        scaling_factor = image.shape[0] / height_before_scale
    return (
        encode_image_to_jpeg_bytes(image=image, jpeg_quality=jpeg_compression_level),
        scaling_factor,
    )

def safe_register_image_at_roboflow(
    cache: BaseCache,
    strategy_with_spare_credit: str,
    encoded_image: bytes,
    local_image_id: str,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    tags: List[str],
    inference_id: Optional[str],
) -> Optional[str]:
    credit_to_be_returned = False
    try:
        registration_response = register_image_at_roboflow(
            api_key=api_key,
            dataset_id=configuration.dataset_id,
            local_image_id=local_image_id,
            image_bytes=encoded_image,
            batch_name=batch_name,
            tags=tags,
            inference_id=inference_id,
        )
        image_duplicated = registration_response.get("duplicate", False)
        if image_duplicated:
            credit_to_be_returned = True
            logger.warning(f"Image duplication detected: {registration_response}.")
            return None
        return registration_response["id"]
    except Exception as error:
        credit_to_be_returned = True
        raise error
    finally:
        if credit_to_be_returned:
            return_strategy_credit(
                cache=cache,
                workspace=configuration.workspace_id,
                project=configuration.dataset_id,
                strategy_name=strategy_with_spare_credit,
            )

