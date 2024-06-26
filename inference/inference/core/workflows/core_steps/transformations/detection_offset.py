def offset_detection(
    detection: Dict[str, Any], offset_width: int, offset_height: int
) -> Dict[str, Any]:
    detection_copy = deepcopy(detection)
    detection_copy[WIDTH_KEY] += round(offset_width)
    detection_copy[HEIGHT_KEY] += round(offset_height)
    detection_copy[PARENT_ID_KEY] = detection_copy[DETECTION_ID_KEY]
    detection_copy[DETECTION_ID_KEY] = str(uuid4())
    return detection_copy