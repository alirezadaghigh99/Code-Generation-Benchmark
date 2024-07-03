def join_cameras_as_batch(cameras_list: Sequence[CamerasBase]) -> CamerasBase:
    """
    Create a batched cameras object by concatenating a list of input
    cameras objects. All the tensor attributes will be joined along
    the batch dimension.

    Args:
        cameras_list: List of camera classes all of the same type and
            on the same device. Each represents one or more cameras.
    Returns:
        cameras: single batched cameras object of the same
            type as all the objects in the input list.
    """
    # Get the type and fields to join from the first camera in the batch
    c0 = cameras_list[0]
    fields = c0._FIELDS
    shared_fields = c0._SHARED_FIELDS

    if not all(isinstance(c, CamerasBase) for c in cameras_list):
        raise ValueError("cameras in cameras_list must inherit from CamerasBase")

    if not all(type(c) is type(c0) for c in cameras_list[1:]):
        raise ValueError("All cameras must be of the same type")

    if not all(c.device == c0.device for c in cameras_list[1:]):
        raise ValueError("All cameras in the batch must be on the same device")

    # Concat the fields to make a batched tensor
    kwargs = {}
    kwargs["device"] = c0.device

    for field in fields:
        field_not_none = [(getattr(c, field) is not None) for c in cameras_list]
        if not any(field_not_none):
            continue
        if not all(field_not_none):
            raise ValueError(f"Attribute {field} is inconsistently present")

        attrs_list = [getattr(c, field) for c in cameras_list]

        if field in shared_fields:
            # Only needs to be set once
            if not all(a == attrs_list[0] for a in attrs_list):
                raise ValueError(f"Attribute {field} is not constant across inputs")

            # e.g. "in_ndc" is set as attribute "_in_ndc" on the class
            # but provided as "in_ndc" in the input args
            if field.startswith("_"):
                field = field[1:]

            kwargs[field] = attrs_list[0]
        elif isinstance(attrs_list[0], torch.Tensor):
            # In the init, all inputs will be converted to
            # batched tensors before set as attributes
            # Join as a tensor along the batch dimension
            kwargs[field] = torch.cat(attrs_list, dim=0)
        else:
            raise ValueError(f"Field {field} type is not supported for batching")

    return c0.__class__(**kwargs)