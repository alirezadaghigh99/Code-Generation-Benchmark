def is_prediction_registration_forbidden(
    prediction: Optional[Union[sv.Detections, dict]],
) -> bool:
    if prediction is None:
        return True
    if isinstance(prediction, sv.Detections) and len(prediction) == 0:
        return True
    if isinstance(prediction, dict) and "top" not in prediction:
        return True
    return False

def register_datapoint(
    target_project: str,
    encoded_image: bytes,
    local_image_id: str,
    prediction: Optional[Union[sv.Detections, dict]],
    api_key: str,
    batch_name: str,
    tags: List[str],
) -> str:
    roboflow_image_id = safe_register_image_at_roboflow(
        target_project=target_project,
        encoded_image=encoded_image,
        local_image_id=local_image_id,
        api_key=api_key,
        batch_name=batch_name,
        tags=tags,
    )
    if roboflow_image_id is None:
        return DUPLICATED_STATUS
    if is_prediction_registration_forbidden(prediction=prediction):
        return "Successfully registered image"
    encoded_prediction, prediction_format = encode_prediction(prediction=prediction)
    _ = annotate_image_at_roboflow(
        api_key=api_key,
        dataset_id=target_project,
        local_image_id=local_image_id,
        roboflow_image_id=roboflow_image_id,
        annotation_content=encoded_prediction,
        annotation_file_type=prediction_format,
        is_prediction=True,
    )
    return "Successfully registered image and annotation"

def encode_prediction(
    prediction: Union[sv.Detections, dict],
) -> Tuple[str, str]:
    if isinstance(prediction, dict):
        return prediction["top"], "txt"
    detections_in_inference_format = serialise_sv_detections(detections=prediction)
    return json.dumps(detections_in_inference_format), "json"

