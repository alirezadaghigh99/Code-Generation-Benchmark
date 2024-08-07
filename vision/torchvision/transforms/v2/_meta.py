class ConvertBoundingBoxFormat(Transform):
    """Convert bounding box coordinates to the given ``format``, eg from "CXCYWH" to "XYXY".

    Args:
        format (str or tv_tensors.BoundingBoxFormat): output bounding box format.
            Possible values are defined by :class:`~torchvision.tv_tensors.BoundingBoxFormat` and
            string values match the enums, e.g. "XYXY" or "XYWH" etc.
    """

    _transformed_types = (tv_tensors.BoundingBoxes,)

    def __init__(self, format: Union[str, tv_tensors.BoundingBoxFormat]) -> None:
        super().__init__()
        self.format = format

    def _transform(self, inpt: tv_tensors.BoundingBoxes, params: Dict[str, Any]) -> tv_tensors.BoundingBoxes:
        return F.convert_bounding_box_format(inpt, new_format=self.format)  # type: ignore[return-value, arg-type]

