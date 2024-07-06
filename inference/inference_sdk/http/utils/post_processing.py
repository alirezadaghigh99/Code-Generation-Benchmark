def transform_base64_visualisation(
    visualisation: str,
    expected_format: VisualisationResponseFormat,
) -> Union[str, np.ndarray, Image.Image]:
    visualisation_bytes = base64.b64decode(visualisation)
    return transform_visualisation_bytes(
        visualisation=visualisation_bytes, expected_format=expected_format
    )

def response_contains_jpeg_image(response: Response) -> bool:
    content_type = None
    for header_name in CONTENT_TYPE_HEADERS:
        if header_name in response.headers:
            content_type = response.headers[header_name]
            break
    if content_type is None:
        return False
    return "image/jpeg" in content_type

def decode_workflow_output_image(
    value: Dict[str, Any],
    expected_format: VisualisationResponseFormat,
) -> Union[str, np.ndarray, Image.Image]:
    if expected_format is VisualisationResponseFormat.BASE64:
        return value["value"]
    return transform_base64_visualisation(
        visualisation=value["value"],
        expected_format=expected_format,
    )

def is_workflow_image(value: Any) -> bool:
    return issubclass(type(value), dict) and value.get("type") == "base64"

def combine_gaze_detections(
    detections: Union[dict, List[Union[dict, List[dict]]]]
) -> Union[dict, List[Dict]]:
    if not issubclass(type(detections), list):
        return detections
    detections = [e if issubclass(type(e), list) else [e] for e in detections]
    return list(itertools.chain.from_iterable(detections))

def combine_clip_embeddings(embeddings: Union[dict, List[dict]]) -> List[dict]:
    if issubclass(type(embeddings), list):
        result = []
        for e in embeddings:
            result.extend(combine_clip_embeddings(embeddings=e))
        return result
    frame_id = embeddings["frame_id"]
    time = embeddings["time"]
    if len(embeddings["embeddings"]) > 1:
        new_embeddings = [
            {"frame_id": frame_id, "time": time, "embeddings": [e]}
            for e in embeddings["embeddings"]
        ]
    else:
        new_embeddings = [embeddings]
    return new_embeddings

def filter_model_descriptions(
    descriptions: List[ModelDescription],
    model_id: str,
) -> Optional[ModelDescription]:
    matching_models = [d for d in descriptions if d.model_id == model_id]
    if len(matching_models) > 0:
        return matching_models[0]
    return None

