class SanitizeBoundingBoxes(Transform):
    """Remove degenerate/invalid bounding boxes and their corresponding labels and masks.

    This transform removes bounding boxes and their associated labels/masks that:

    - are below a given ``min_size`` or ``min_area``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :class:`~torchvision.transforms.v2.ClampBoundingBoxes` first to avoid undesired removals.

    It can also sanitize other tensors like the "iscrowd" or "area" properties from COCO
    (see ``labels_getter`` parameter).

    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`~torchvision.transforms.v2.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        min_size (float, optional): The size below which bounding boxes are removed. Default is 1.
        min_area (float, optional): The area below which bounding boxes are removed. Default is 1.
        labels_getter (callable or str or None, optional): indicates how to identify the labels in the input
            (or anything else that needs to be sanitized along with the bounding boxes).
            By default, this will try to find a "labels" key in the input (case-insensitive), if
            the input is a dict or it is a tuple whose second element is a dict.
            This heuristic should work well with a lot of datasets, including the built-in torchvision datasets.

            It can also be a callable that takes the same input as the transform, and returns either:

            - A single tensor (the labels)
            - A tuple/list of tensors, each of which will be subject to the same sanitization as the bounding boxes.
              This is useful to sanitize multiple tensors like the labels, and the "iscrowd" or "area" properties
              from COCO.

            If ``labels_getter`` is None then only bounding boxes are sanitized.
    """

    def __init__(
        self,
        min_size: float = 1.0,
        min_area: float = 1.0,
        labels_getter: Union[Callable[[Any], Any], str, None] = "default",
    ) -> None:
        super().__init__()

        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}.")
        self.min_size = min_size

        if min_area < 1:
            raise ValueError(f"min_area must be >= 1, got {min_area}.")
        self.min_area = min_area

        self.labels_getter = labels_getter
        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs: Any) -> Any:
        inputs = inputs if len(inputs) > 1 else inputs[0]

        labels = self._labels_getter(inputs)
        if labels is not None:
            msg = "The labels in the input to forward() must be a tensor or None, got {type} instead."
            if isinstance(labels, torch.Tensor):
                labels = (labels,)
            elif isinstance(labels, (tuple, list)):
                for entry in labels:
                    if not isinstance(entry, torch.Tensor):
                        # TODO: we don't need to enforce tensors, just that entries are indexable as t[bool_mask]
                        raise ValueError(msg.format(type=type(entry)))
            else:
                raise ValueError(msg.format(type=type(labels)))

        flat_inputs, spec = tree_flatten(inputs)
        boxes = get_bounding_boxes(flat_inputs)

        if labels is not None:
            for label in labels:
                if boxes.shape[0] != label.shape[0]:
                    raise ValueError(
                        f"Number of boxes (shape={boxes.shape}) and must match the number of labels."
                        f"Found labels with shape={label.shape})."
                    )

        valid = F._misc._get_sanitize_bounding_boxes_mask(
            boxes,
            format=boxes.format,
            canvas_size=boxes.canvas_size,
            min_size=self.min_size,
            min_area=self.min_area,
        )

        params = dict(valid=valid, labels=labels)
        flat_outputs = [self._transform(inpt, params) for inpt in flat_inputs]

        return tree_unflatten(flat_outputs, spec)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = params["labels"] is not None and any(inpt is label for label in params["labels"])
        is_bounding_boxes_or_mask = isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask))

        if not (is_label or is_bounding_boxes_or_mask):
            return inpt

        output = inpt[params["valid"]]

        if is_label:
            return output
        else:
            return tv_tensors.wrap(output, like=inpt)

class Identity(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

