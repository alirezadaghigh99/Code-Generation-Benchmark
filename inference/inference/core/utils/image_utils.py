def load_image_rgb(value: Any, disable_preproc_auto_orient: bool = False) -> np.ndarray:
    np_image, is_bgr = load_image(
        value=value, disable_preproc_auto_orient=disable_preproc_auto_orient
    )
    if is_bgr:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return np_image