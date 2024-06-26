def register_box(name: str,
                 box_type: Type = None,
                 force: bool = False) -> Union[Type, Callable]:
    """Register a box type.

    A record will be added to ``bbox_types``, whose key is the box type name
    and value is the box type itself. Simultaneously, a reverse dictionary
    ``_box_type_to_name`` will be updated. It can be used as a decorator or
    a normal function.

    Args:
        name (str): The name of box type.
        bbox_type (type, Optional): Box type class to be registered.
            Defaults to None.
        force (bool): Whether to override the existing box type with the same
            name. Defaults to False.

    Examples:
        >>> from mmdet.structures.bbox import register_box
        >>> from mmdet.structures.bbox import BaseBoxes

        >>> # as a decorator
        >>> @register_box('hbox')
        >>> class HorizontalBoxes(BaseBoxes):
        >>>     pass

        >>> # as a normal function
        >>> class RotatedBoxes(BaseBoxes):
        >>>     pass
        >>> register_box('rbox', RotatedBoxes)
    """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method: register_box(name, box_type=BoxCls)
    if box_type is not None:
        _register_box(name=name, box_type=box_type, force=force)
        return box_type

    # use it as a decorator: @register_box(name)
    def _register(cls):
        _register_box(name=name, box_type=cls, force=force)
        return cls

    return _register