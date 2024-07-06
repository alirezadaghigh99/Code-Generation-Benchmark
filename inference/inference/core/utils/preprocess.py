def static_crop_should_be_applied(
    preprocessing_config: dict,
    disable_preproc_static_crop: bool,
) -> bool:
    return (
        STATIC_CROP_KEY in preprocessing_config.keys()
        and not DISABLE_PREPROC_STATIC_CROP
        and not disable_preproc_static_crop
        and preprocessing_config[STATIC_CROP_KEY][ENABLED_KEY]
    )

def contrast_adjustments_should_be_applied(
    preprocessing_config: dict,
    disable_preproc_contrast: bool,
) -> bool:
    return (
        CONTRAST_KEY in preprocessing_config.keys()
        and not DISABLE_PREPROC_CONTRAST
        and not disable_preproc_contrast
        and preprocessing_config[CONTRAST_KEY][ENABLED_KEY]
    )

def grayscale_conversion_should_be_applied(
    preprocessing_config: dict,
    disable_preproc_grayscale: bool,
) -> bool:
    return (
        GRAYSCALE_KEY in preprocessing_config.keys()
        and not DISABLE_PREPROC_GRAYSCALE
        and not disable_preproc_grayscale
        and preprocessing_config[GRAYSCALE_KEY][ENABLED_KEY]
    )

def take_static_crop(image: np.ndarray, crop_parameters: Dict[str, int]) -> np.ndarray:
    height, width = image.shape[0:2]
    x_min = int(crop_parameters["x_min"] / 100 * width)
    y_min = int(crop_parameters["y_min"] / 100 * height)
    x_max = int(crop_parameters["x_max"] / 100 * width)
    y_max = int(crop_parameters["y_max"] / 100 * height)
    return image[y_min:y_max, x_min:x_max, :]

