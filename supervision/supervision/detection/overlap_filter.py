def mask_non_max_suppression(
    predictions: np.ndarray,
    masks: np.ndarray,
    iou_threshold: float = 0.5,
    mask_dimension: int = 640,
) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on segmentation predictions.

    Args:
        predictions (np.ndarray): A 2D array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`. Shape: `(N, 5)` or
            `(N, 6)`, where N is the number of predictions.
        masks (np.ndarray): A 3D array of binary masks corresponding to the predictions.
            Shape: `(N, H, W)`, where N is the number of predictions, and H, W are the
            dimensions of each mask.
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression.
        mask_dimension (int, optional): The dimension to which the masks should be
            resized before computing IOU values. Defaults to 640.

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after
            non-maximum suppression.

    Raises:
        AssertionError: If `iou_threshold` is not within the closed
        range from `0` to `1`.
    """
    assert 0 <= iou_threshold <= 1, (
        "Value of `iou_threshold` must be in the closed range from 0 to 1, "
        f"{iou_threshold} given."
    )
    rows, columns = predictions.shape

    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    sort_index = predictions[:, 4].argsort()[::-1]
    predictions = predictions[sort_index]
    masks = masks[sort_index]
    masks_resized = resize_masks(masks, mask_dimension)
    ious = mask_iou_batch(masks_resized, masks_resized)
    categories = predictions[:, 5]

    keep = np.ones(rows, dtype=bool)
    for i in range(rows):
        if keep[i]:
            condition = (ious[i] > iou_threshold) & (categories[i] == categories)
            keep[i + 1 :] = np.where(condition[i + 1 :], False, keep[i + 1 :])

    return keep[sort_index.argsort()]