def get_box_type(box_type: Union[str, type]) -> Tuple[str, type]:
    """get both box type name and class.

    Args:
        box_type (str or type): Single box type name or class.

    Returns:
        Tuple[str, type]: A tuple of box type name and class.
    """
    if isinstance(box_type, str):
        type_name = box_type.lower()
        assert type_name in box_types, \
            f"Box type {type_name} hasn't been registered in box_types."
        type_cls = box_types[type_name]
    elif issubclass(box_type, BaseBoxes):
        assert box_type in _box_type_to_name, \
            f"Box type {box_type} hasn't been registered in box_types."
        type_name = _box_type_to_name[box_type]
        type_cls = box_type
    else:
        raise KeyError('box_type must be a str or class inheriting from '
                       f'BaseBoxes, but got {type(box_type)}.')
    return type_name, type_cls