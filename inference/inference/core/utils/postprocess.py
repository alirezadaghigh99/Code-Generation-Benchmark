def post_process_keypoints(
    predictions: List[List[List[float]]],
    keypoints_start_index: int,
    infer_shape: Tuple[int, int],
    img_dims: List[Tuple[int, int]],
    preproc: dict,
    disable_preproc_static_crop: bool = False,
    resize_method: str = "Stretch to",
) -> List[List[List[float]]]:
    """Scales and shifts keypoints based on the given image shapes and preprocessing method.

    This function performs polygon scaling and shifting based on the specified resizing method and
    pre-processing steps. The polygons are transformed according to the ratio and padding between two images.

    Args:
        predictions: predictions from model
        keypoints_start_index: offset in the 3rd dimension pointing where in the prediction start keypoints [(x, y, cfg), ...] for each keypoint class
        img_dims list of (tuple of int): Shape of the source image (height, width).
        infer_shape (tuple of int): Shape of the target image (height, width).
        preproc (object): Preprocessing details used for generating the transformation.
        resize_method (str, optional): Resizing method, either "Stretch to", "Fit (black edges) in", "Fit (white edges) in", or "Fit (grey edges) in". Defaults to "Stretch to".
        disable_preproc_static_crop: flag to disable static crop
    Returns:
        list of list of list: predictions with post-processed keypoints
    """
    # Get static crop params
    scaled_predictions = []
    # Loop through batches
    for i, batch_predictions in enumerate(predictions):
        if len(batch_predictions) == 0:
            scaled_predictions.append([])
            continue
        np_batch_predictions = np.array(batch_predictions)
        keypoints = np_batch_predictions[:, keypoints_start_index:]
        (crop_shift_x, crop_shift_y), origin_shape = get_static_crop_dimensions(
            img_dims[i],
            preproc,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        if resize_method == "Stretch to":
            keypoints = stretch_keypoints(
                keypoints=keypoints,
                infer_shape=infer_shape,
                origin_shape=origin_shape,
            )
        elif (
            resize_method == "Fit (black edges) in"
            or resize_method == "Fit (white edges) in"
            or resize_method == "Fit (grey edges) in"
        ):
            keypoints = undo_image_padding_for_predicted_keypoints(
                keypoints=keypoints,
                infer_shape=infer_shape,
                origin_shape=origin_shape,
            )
        keypoints = clip_keypoints_coordinates(
            keypoints=keypoints, origin_shape=origin_shape
        )
        keypoints = shift_keypoints(
            keypoints=keypoints, shift_x=crop_shift_x, shift_y=crop_shift_y
        )
        np_batch_predictions[:, keypoints_start_index:] = keypoints
        scaled_predictions.append(np_batch_predictions.tolist())
    return scaled_predictions