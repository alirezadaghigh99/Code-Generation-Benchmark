def load_image_with_inferred_type(
    value: Any,
    cv_imread_flags: int = cv2.IMREAD_COLOR,
) -> Tuple[np.ndarray, bool]:
    """Load an image by inferring its type.

    Args:
        value (Any): The image data.
        cv_imread_flags (int): Flags used for OpenCV's imread function.

    Returns:
        Tuple[np.ndarray, bool]: Loaded image as a numpy array and a boolean indicating if the image is in BGR format.

    Raises:
        NotImplementedError: If the image type could not be inferred.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        validate_numpy_image(data=value)
        return value, True
    elif isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB")), False
    elif isinstance(value, str) and (value.startswith("http")):
        return load_image_from_url(value=value, cv_imread_flags=cv_imread_flags), True
    elif isinstance(value, str) and os.path.isfile(value):
        return cv2.imread(value, cv_imread_flags), True
    else:
        return attempt_loading_image_from_string(
            value=value, cv_imread_flags=cv_imread_flags
        )