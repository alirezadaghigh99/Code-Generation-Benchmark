def check_bboxes(bboxes: Sequence[BoxType]) -> None:
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for bbox in bboxes:
        check_bbox(bbox)

def filter_bboxes(
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

def normalize_bbox(bbox: BoxType, rows: int, cols: int) -> BoxType:
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.

    Args:
        bbox: Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Returns:
        Normalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """
    if rows <= 0:
        msg = "Argument rows must be positive integer"
        raise ValueError(msg)
    if cols <= 0:
        msg = "Argument cols must be positive integer"
        raise ValueError(msg)

    tail: tuple[Any, ...]
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])
    x_min /= cols
    x_max /= cols
    y_min /= rows
    y_max /= rows

    return cast(BoxType, (x_min, y_min, x_max, y_max, *tail))

def denormalize_bbox(bbox: BoxType, rows: int, cols: int) -> BoxType:
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.

    Args:
        bbox: Normalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Returns:
        Denormalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """
    tail: tuple[Any, ...]
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    if rows <= 0:
        msg = "Argument rows must be positive integer"
        raise ValueError(msg)
    if cols <= 0:
        msg = "Argument cols must be positive integer"
        raise ValueError(msg)

    x_min, x_max = x_min * cols, x_max * cols
    y_min, y_max = y_min * rows, y_max * rows

    return cast(BoxType, (x_min, y_min, x_max, y_max, *tail))

def convert_bbox_to_albumentations(
    bbox: BoxType,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
) -> BoxType:
    """Convert a bounding box from a format specified in `source_format` to the format used by albumentations:
    normalized coordinates of top-left and bottom-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

    Args:
        bbox: A bounding box tuple.
        source_format: format of the bounding box. Should be 'coco', 'pascal_voc', or 'yolo'.
        check_validity: Check if all boxes are valid boxes.
        rows: Image height.
        cols: Image width.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    Note:
        The `coco` format of a bounding box looks like `(x_min, y_min, width, height)`, e.g. (97, 12, 150, 200).
        The `pascal_voc` format of a bounding box looks like `(x_min, y_min, x_max, y_max)`, e.g. (97, 12, 247, 212).
        The `yolo` format of a bounding box looks like `(x, y, width, height)`, e.g. (0.3, 0.1, 0.05, 0.07);
        where `x`, `y` coordinates of the center of the box, all values normalized to 1 by image height and width.

    Raises:
        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`, or `yolo`.
        ValueError: If in YOLO format all labels not in range (0, 1).

    """
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
        )

    if source_format == "coco":
        (x_min, y_min, width, height), tail = bbox[:4], bbox[4:]
        x_max = x_min + width
        y_max = y_min + height
    elif source_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12
        _bbox = np.array(bbox[:4])
        if check_validity and np.any((_bbox <= 0) | (_bbox > 1)):
            msg = "In YOLO format all coordinates must be float and in range (0, 1]"
            raise ValueError(msg)

        (x, y, width, height), tail = bbox[:4], bbox[4:]

        w_half, h_half = width / 2, height / 2
        x_min = x - w_half
        y_min = y - h_half
        x_max = x_min + width
        y_max = y_min + height
    else:
        (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]

    bbox = (x_min, y_min, x_max, y_max, *tuple(tail))

    if source_format != "yolo":
        bbox = normalize_bbox(bbox, rows, cols)
    if check_validity:
        check_bbox(bbox)
    return bbox

def convert_bbox_from_albumentations(
    bbox: BoxType,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
) -> BoxType:
    """Convert a bounding box from the format used by albumentations to a format, specified in `target_format`.

    Args:
        bbox: An albumentations bounding box `(x_min, y_min, x_max, y_max)`.
        target_format: required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows: Image height.
        cols: Image width.
        check_validity: Check if all boxes are valid boxes.

    Returns:
        tuple: A bounding box.

    Note:
        The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
        The `yolo` format of a bounding box looks like `[x, y, width, height]`, e.g. [0.3, 0.1, 0.05, 0.07].

    Raises:
        ValueError: if `target_format` is not equal to `coco`, `pascal_voc` or `yolo`.

    """
    if target_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown target_format {target_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
        )
    if check_validity:
        check_bbox(bbox)

    if target_format != "yolo":
        bbox = denormalize_bbox(bbox, rows, cols)
    if target_format == "coco":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])
        width = x_max - x_min
        height = y_max - y_min
        bbox = cast(BoxType, (x_min, y_min, width, height, *tail))
    elif target_format == "yolo":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        bbox = cast(BoxType, (x, y, width, height, *tail))
    return bbox

class BboxParams(Params):
    """Parameters of bounding boxes

    Args:
        format (str): format of bounding boxes. Should be `coco`, `pascal_voc`, `albumentations` or `yolo`.

            The `coco` format
                `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
            The `pascal_voc` format
                `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
            The `albumentations` format
                is like `pascal_voc`, but normalized,
                in other words: `[x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
            The `yolo` format
                `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
                `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.

        label_fields (list): List of fields joined with boxes, e.g., labels.
        min_area (float): Minimum area of a bounding box in pixels or normalized units.
            Bounding boxes with an area less than this value will be removed. Default: 0.0.
        min_visibility (float): Minimum fraction of area for a bounding box to remain in the list.
            Bounding boxes with a visible area less than this fraction will be removed. Default: 0.0.
        min_width (float): Minimum width of a bounding box in pixels or normalized units.
            Bounding boxes with a width less than this value will be removed. Default: 0.0.
        min_height (float): Minimum height of a bounding box in pixels or normalized units.
            Bounding boxes with a height less than this value will be removed. Default: 0.0.
        check_each_transform (bool): If True, bounding boxes will be checked after each dual transform. Default: True.
        clip (bool): If True, bounding boxes will be clipped to the image borders before applying any transform.
            Default: False.

    """

    def __init__(
        self,
        format: str,  # noqa: A002
        label_fields: Sequence[Any] | None = None,
        min_area: float = 0.0,
        min_visibility: float = 0.0,
        min_width: float = 0.0,
        min_height: float = 0.0,
        check_each_transform: bool = True,
        clip: bool = False,
    ):
        super().__init__(format, label_fields)
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.min_width = min_width
        self.min_height = min_height
        self.check_each_transform = check_each_transform
        self.clip = clip

    def to_dict_private(self) -> dict[str, Any]:
        data = super().to_dict_private()
        data.update(
            {
                "min_area": self.min_area,
                "min_visibility": self.min_visibility,
                "min_width": self.min_width,
                "min_height": self.min_height,
                "check_each_transform": self.check_each_transform,
                "clip": self.clip,
            },
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        return "BboxParams"

