def bytes_to_opencv_image(
    payload: bytes, array_type: np.number = np.uint8
) -> np.ndarray:
    bytes_array = np.frombuffer(payload, dtype=array_type)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    if decoding_result is None:
        raise EncodingError("Could not encode bytes to OpenCV image.")
    return decoding_result

def bytes_to_pillow_image(payload: bytes) -> Image.Image:
    buffer = BytesIO(payload)
    try:
        return Image.open(buffer)
    except UnidentifiedImageError as error:
        raise EncodingError("Could not encode bytes to PIL image.") from error

