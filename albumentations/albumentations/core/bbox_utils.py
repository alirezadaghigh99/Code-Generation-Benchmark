def check_bboxes(bboxes: Sequence[BoxType]) -> None:
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for bbox in bboxes:
        check_bbox(bbox)def filter_bboxes(
    bboxes: Sequence[BoxType],
    rows: int,
    cols: int,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> list[BoxType]:
    """Remove bounding boxes that either lie outside of the visible area by more then min_visibility
    or whose area in pixels is under the threshold set by `min_area`. Also it crops boxes to final image size.

    Args:
        bboxes: list of albumentations bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.
        min_area: Minimum area of a bounding box. All bounding boxes whose visible area in pixels.
            is less than this value will be removed. Default: 0.0.
        min_visibility: Minimum fraction of area for a bounding box to remain this box in list. Default: 0.0.
        min_width: Minimum width of a bounding box. All bounding boxes whose width is
            less than this value will be removed. Default: 0.0.
        min_height: Minimum height of a bounding box. All bounding boxes whose height is
            less than this value will be removed. Default: 0.0.

    Returns:
        list of bounding boxes.

    """
    resulting_boxes: list[BoxType] = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        # Calculate areas of bounding box before and after clipping.
        transformed_box_area = calculate_bbox_area(bbox, rows, cols)
        clipped_bbox = clip_bbox(bbox, rows, cols)

        bbox, tail = clipped_bbox[:4], clipped_bbox[4:]

        clipped_box_area = calculate_bbox_area(bbox, rows, cols)

        # Calculate width and height of the clipped bounding box.
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
        clipped_width, clipped_height = x_max - x_min, y_max - y_min

        if (
            clipped_box_area != 0  # to ensure transformed_box_area!=0 and to handle min_area=0 or min_visibility=0
            and clipped_box_area >= min_area
            and clipped_box_area / transformed_box_area >= min_visibility
            and clipped_width >= min_width
            and clipped_height >= min_height
        ):
            resulting_boxes.append(cast(BoxType, bbox + tail))
    return resulting_boxes