def prepare(
    image: np.ndarray,
    preproc,
    disable_preproc_contrast: bool = False,
    disable_preproc_grayscale: bool = False,
    disable_preproc_static_crop: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Prepares an image by applying a series of preprocessing steps defined in the `preproc` dictionary.

    Args:
        image (PIL.Image.Image): The input PIL image object.
        preproc (dict): Dictionary containing preprocessing steps. Example:
            {
                "resize": {"enabled": true, "width": 416, "height": 416, "format": "Stretch to"},
                "static-crop": {"y_min": 25, "x_max": 75, "y_max": 75, "enabled": true, "x_min": 25},
                "auto-orient": {"enabled": true},
                "grayscale": {"enabled": true},
                "contrast": {"enabled": true, "type": "Adaptive Equalization"}
            }
        disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
        disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
        disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

    Returns:
        PIL.Image.Image: The preprocessed image object.
        tuple: The dimensions of the image.

    Note:
        The function uses global flags like `DISABLE_PREPROC_AUTO_ORIENT`, `DISABLE_PREPROC_STATIC_CROP`, etc.
        to conditionally enable or disable certain preprocessing steps.
    """
    try:
        h, w = image.shape[0:2]
        img_dims = (h, w)
        if static_crop_should_be_applied(
            preprocessing_config=preproc,
            disable_preproc_static_crop=disable_preproc_static_crop,
        ):
            image = take_static_crop(
                image=image, crop_parameters=preproc[STATIC_CROP_KEY]
            )
        if contrast_adjustments_should_be_applied(
            preprocessing_config=preproc,
            disable_preproc_contrast=disable_preproc_contrast,
        ):
            adjustment_type = ContrastAdjustmentType(preproc[CONTRAST_KEY][TYPE_KEY])
            image = apply_contrast_adjustment(
                image=image, adjustment_type=adjustment_type
            )
        if grayscale_conversion_should_be_applied(
            preprocessing_config=preproc,
            disable_preproc_grayscale=disable_preproc_grayscale,
        ):
            image = apply_grayscale_conversion(image=image)
        return image, img_dims
    except KeyError as error:
        raise PreProcessingError(
            f"Pre-processing of image failed due to misconfiguration. Missing key: {error}."
        ) from error