def sv_detections_to_root_coordinates(
    detections: sv.Detections, keypoints_key: str = KEYPOINTS_XY_KEY_IN_SV_DETECTIONS
) -> sv.Detections:
    detections_copy = deepcopy(detections)
    if len(detections_copy) == 0:
        return detections_copy

    if any(
        key not in detections_copy.data
        for key in KEYS_REQUIRED_TO_EMBED_IN_ROOT_COORDINATES
    ):
        logging.warning(
            "Could not execute detections_to_root_coordinates(...) on detections with "
            f"the following metadata registered: {list(detections_copy.data.keys())}"
        )
        return detections_copy
    if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in detections_copy.data:
        scale = detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY][0]
        detections_copy = scale_sv_detections(
            detections=detections,
            scale=1 / scale,
        )
    detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = np.array(
        [1.0] * len(detections_copy)
    )
    detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array(
        [1.0] * len(detections_copy)
    )
    origin_height = detections_copy[ROOT_PARENT_DIMENSIONS_KEY][0][0]
    origin_width = detections_copy[ROOT_PARENT_DIMENSIONS_KEY][0][1]
    detections_copy[IMAGE_DIMENSIONS_KEY] = np.array(
        [[origin_height, origin_width]] * len(detections_copy)
    )
    root_parent_id = detections_copy[ROOT_PARENT_ID_KEY][0]
    shift_x, shift_y = detections_copy[ROOT_PARENT_COORDINATES_KEY][0]
    detections_copy.xyxy += [shift_x, shift_y, shift_x, shift_y]
    if keypoints_key in detections_copy.data:
        for keypoints in detections_copy[keypoints_key]:
            if len(keypoints):
                keypoints += [shift_x, shift_y]
    if detections_copy.mask is not None:
        origin_mask_base = np.full((origin_height, origin_width), False)
        new_anchored_masks = np.array(
            [origin_mask_base.copy() for _ in detections_copy]
        )
        for anchored_mask, original_mask in zip(
            new_anchored_masks, detections_copy.mask
        ):
            mask_h, mask_w = original_mask.shape
            # TODO: instead of shifting mask we could store contours in data instead of storing mask (even if calculated)
            #       it would be faster to shift contours but at expense of having to remember to generate mask from contour when it's needed
            anchored_mask[shift_y : shift_y + mask_h, shift_x : shift_x + mask_w] = (
                original_mask
            )
        detections_copy.mask = new_anchored_masks
    new_root_metadata = ImageParentMetadata(
        parent_id=root_parent_id,
        origin_coordinates=OriginCoordinatesSystem(
            left_top_y=0,
            left_top_x=0,
            origin_width=origin_width,
            origin_height=origin_height,
        ),
    )
    detections_copy = attach_parent_coordinates_to_detections(
        detections=detections_copy,
        parent_metadata=new_root_metadata,
        parent_id_key=ROOT_PARENT_ID_KEY,
        coordinates_key=ROOT_PARENT_COORDINATES_KEY,
        dimensions_key=ROOT_PARENT_DIMENSIONS_KEY,
    )
    return attach_parent_coordinates_to_detections(
        detections=detections_copy,
        parent_metadata=new_root_metadata,
        parent_id_key=PARENT_ID_KEY,
        coordinates_key=PARENT_COORDINATES_KEY,
        dimensions_key=PARENT_DIMENSIONS_KEY,
    )

def filter_out_unwanted_classes_from_sv_detections_batch(
    predictions: List[sv.Detections],
    classes_to_accept: Optional[List[str]],
) -> List[sv.Detections]:
    if not classes_to_accept:
        return predictions
    filtered_predictions = []
    for prediction in predictions:
        filtered_prediction = prediction[
            np.isin(prediction[CLASS_NAME_DATA_FIELD], classes_to_accept)
        ]
        filtered_predictions.append(filtered_prediction)
    return filtered_predictions

def scale_sv_detections(
    detections: sv.Detections,
    scale: float,
    keypoints_key: str = KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
) -> sv.Detections:
    detections_copy = deepcopy(detections)
    if len(detections_copy) == 0:
        return detections_copy
    detections_copy.xyxy = (detections_copy.xyxy * scale).round()
    if keypoints_key in detections_copy.data:
        for i in range(len(detections_copy[keypoints_key])):
            detections_copy[keypoints_key][i] = (
                detections_copy[keypoints_key][i].astype(np.float32) * scale
            ).round()
    detections_copy[IMAGE_DIMENSIONS_KEY] = (
        detections_copy[IMAGE_DIMENSIONS_KEY] * scale
    ).round()
    if detections_copy.mask is not None:
        scaled_masks = []
        original_mask_size_wh = (
            detections_copy.mask.shape[2],
            detections_copy.mask.shape[1],
        )
        scaled_mask_size_wh = round(original_mask_size_wh[0] * scale), round(
            original_mask_size_wh[1] * scale
        )
        for detection_mask in detections_copy.mask:
            polygons = sv.mask_to_polygons(mask=detection_mask)
            polygon_masks = []
            for polygon in polygons:
                scaled_polygon = (polygon * scale).round().astype(np.int32)
                polygon_masks.append(
                    sv.polygon_to_mask(
                        polygon=scaled_polygon, resolution_wh=scaled_mask_size_wh
                    )
                )
            scaled_detection_mask = np.sum(polygon_masks, axis=0) > 0
            scaled_masks.append(scaled_detection_mask)
        detections_copy.mask = np.array(scaled_masks)
    if SCALING_RELATIVE_TO_PARENT_KEY in detections_copy.data:
        detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = (
            detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] * scale
        )
    else:
        detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = np.array(
            [scale] * len(detections_copy)
        )
    if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in detections_copy.data:
        detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = (
            detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] * scale
        )
    else:
        detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array(
            [scale] * len(detections_copy)
        )
    return detections_copy

def remove_unexpected_keys_from_dictionary(
    dictionary: dict,
    expected_keys: set,
) -> dict:
    """This function mutates input `dictionary`"""
    unexpected_keys = set(dictionary.keys()).difference(expected_keys)
    for unexpected_key in unexpected_keys:
        del dictionary[unexpected_key]
    return dictionary

def grab_batch_parameters(
    operations_parameters: Dict[str, Any],
    main_batch_size: int,
) -> Dict[str, Any]:
    return {
        key: value.broadcast(n=main_batch_size)
        for key, value in operations_parameters.items()
        if isinstance(value, Batch)
    }

def attach_prediction_type_info(
    predictions: List[Dict[str, Any]],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[Dict[str, Any]]:
    for result in predictions:
        result[key] = prediction_type
    return predictions

def attach_prediction_type_info_to_sv_detections_batch(
    predictions: List[sv.Detections],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[sv.Detections]:
    for prediction in predictions:
        prediction[key] = np.array([prediction_type] * len(prediction))
    return predictions

def grab_non_batch_parameters(operations_parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in operations_parameters.items()
        if not isinstance(value, Batch)
    }

