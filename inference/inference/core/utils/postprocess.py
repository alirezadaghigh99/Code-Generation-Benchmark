def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Union[np.number, np.ndarray]:
    """
    Compute the cosine similarity between two vectors.

    Args:
        a (np.ndarray): Vector A.
        b (np.ndarray): Vector B.

    Returns:
        float: Cosine similarity between vectors A and B.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_static_crop_dimensions(
    orig_shape: Tuple[int, int],
    preproc: dict,
    disable_preproc_static_crop: bool = False,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Generates a transformation based on preprocessing configuration.

    Args:
        orig_shape (tuple): The original shape of the object (e.g., image) - (height, width).
        preproc (dict): Preprocessing configuration dictionary, containing information such as static cropping.
        disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

    Returns:
        tuple: A tuple containing the shift in the x and y directions, and the updated original shape after cropping.
    """
    try:
        if static_crop_should_be_applied(
            preprocessing_config=preproc,
            disable_preproc_static_crop=disable_preproc_static_crop,
        ):
            x_min, y_min, x_max, y_max = standardise_static_crop(
                static_crop_config=preproc[STATIC_CROP_KEY]
            )
        else:
            x_min, y_min, x_max, y_max = 0, 0, 1, 1
        crop_shift_x, crop_shift_y = (
            round(x_min * orig_shape[1]),
            round(y_min * orig_shape[0]),
        )
        cropped_percent_x = x_max - x_min
        cropped_percent_y = y_max - y_min
        orig_shape = (
            round(orig_shape[0] * cropped_percent_y),
            round(orig_shape[1] * cropped_percent_x),
        )
        return (crop_shift_x, crop_shift_y), orig_shape
    except KeyError as error:
        raise PostProcessingError(
            f"Could not find a proper configuration key {error} in post-processing."
        )

def post_process_polygons(
    origin_shape: Tuple[int, int],
    polys: List[List[Tuple[float, float]]],
    infer_shape: Tuple[int, int],
    preproc: dict,
    resize_method: str = "Stretch to",
) -> List[List[Tuple[float, float]]]:
    """Scales and shifts polygons based on the given image shapes and preprocessing method.

    This function performs polygon scaling and shifting based on the specified resizing method and
    pre-processing steps. The polygons are transformed according to the ratio and padding between two images.

    Args:
        origin_shape (tuple of int): Shape of the source image (height, width).
        infer_shape (tuple of int): Shape of the target image (height, width).
        polys (list of list of tuple): List of polygons, where each polygon is represented by a list of (x, y) coordinates.
        preproc (object): Preprocessing details used for generating the transformation.
        resize_method (str, optional): Resizing method, either "Stretch to", "Fit (black edges) in", "Fit (white edges) in", or "Fit (grey edges) in". Defaults to "Stretch to".

    Returns:
        list of list of tuple: A list of shifted and scaled polygons.
    """
    (crop_shift_x, crop_shift_y), origin_shape = get_static_crop_dimensions(
        origin_shape, preproc
    )
    new_polys = []
    if resize_method == "Stretch to":
        width_ratio = origin_shape[1] / infer_shape[1]
        height_ratio = origin_shape[0] / infer_shape[0]
        new_polys = scale_polygons(
            polygons=polys,
            x_scale=width_ratio,
            y_scale=height_ratio,
        )
    elif resize_method in {
        "Fit (black edges) in",
        "Fit (white edges) in",
        "Fit (grey edges) in",
    }:
        new_polys = undo_image_padding_for_predicted_polygons(
            polygons=polys,
            infer_shape=infer_shape,
            origin_shape=origin_shape,
        )
    shifted_polys = []
    for poly in new_polys:
        poly = [(p[0] + crop_shift_x, p[1] + crop_shift_y) for p in poly]
        shifted_polys.append(poly)
    return shifted_polys

def stretch_bboxes(
    predicted_bboxes: np.ndarray,
    infer_shape: Tuple[int, int],
    origin_shape: Tuple[int, int],
) -> np.ndarray:
    scale_height = origin_shape[0] / infer_shape[0]
    scale_width = origin_shape[1] / infer_shape[1]
    return scale_bboxes(
        bboxes=predicted_bboxes,
        scale_x=scale_width,
        scale_y=scale_height,
    )

def post_process_bboxes(
    predictions: List[List[List[float]]],
    infer_shape: Tuple[int, int],
    img_dims: List[Tuple[int, int]],
    preproc: dict,
    disable_preproc_static_crop: bool = False,
    resize_method: str = "Stretch to",
) -> List[List[List[float]]]:
    """
    Postprocesses each patch of detections by scaling them to the original image coordinates and by shifting them based on a static crop preproc (if applied).

    Args:
        predictions (List[List[List[float]]]): The predictions output from NMS, indices are: batch x prediction x [x1, y1, x2, y2, ...].
        infer_shape (Tuple[int, int]): The shape of the inference image.
        img_dims (List[Tuple[int, int]]): The dimensions of the original image for each batch, indices are: batch x [height, width].
        preproc (dict): Preprocessing configuration dictionary.
        disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
        resize_method (str, optional): Resize method for image. Defaults to "Stretch to".

    Returns:
        List[List[List[float]]]: The scaled and shifted predictions, indices are: batch x prediction x [x1, y1, x2, y2, ...].
    """

    # Get static crop params
    scaled_predictions = []
    # Loop through batches
    for i, batch_predictions in enumerate(predictions):
        if len(batch_predictions) == 0:
            scaled_predictions.append([])
            continue
        np_batch_predictions = np.array(batch_predictions)
        # Get bboxes from predictions (x1,y1,x2,y2)
        predicted_bboxes = np_batch_predictions[:, :4]
        (crop_shift_x, crop_shift_y), origin_shape = get_static_crop_dimensions(
            img_dims[i],
            preproc,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        if resize_method == "Stretch to":
            predicted_bboxes = stretch_bboxes(
                predicted_bboxes=predicted_bboxes,
                infer_shape=infer_shape,
                origin_shape=origin_shape,
            )
        elif (
            resize_method == "Fit (black edges) in"
            or resize_method == "Fit (white edges) in"
            or resize_method == "Fit (grey edges) in"
        ):
            predicted_bboxes = undo_image_padding_for_predicted_boxes(
                predicted_bboxes=predicted_bboxes,
                infer_shape=infer_shape,
                origin_shape=origin_shape,
            )
        predicted_bboxes = clip_boxes_coordinates(
            predicted_bboxes=predicted_bboxes,
            origin_shape=origin_shape,
        )
        predicted_bboxes = shift_bboxes(
            bboxes=predicted_bboxes,
            shift_x=crop_shift_x,
            shift_y=crop_shift_y,
        )
        np_batch_predictions[:, :4] = predicted_bboxes
        scaled_predictions.append(np_batch_predictions.tolist())
    return scaled_predictions

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

