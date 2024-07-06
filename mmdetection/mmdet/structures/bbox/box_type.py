def register_box_converter(src_type: Union[str, type],
                           dst_type: Union[str, type],
                           converter: Optional[Callable] = None,
                           force: bool = False) -> Callable:
    """Register a box converter.

    A record will be added to ``box_converter``, whose key is
    '{src_type_name}2{dst_type_name}' and value is the convert function.
    It can be used as a decorator or a normal function.

    Args:
        src_type (str or type): source box type name or class.
        dst_type (str or type): destination box type name or class.
        converter (Callable): Convert function. Defaults to None.
        force (bool): Whether to override the existing box type with the same
            name. Defaults to False.

    Examples:
        >>> from mmdet.structures.bbox import register_box_converter
        >>> # as a decorator
        >>> @register_box_converter('hbox', 'rbox')
        >>> def converter_A(boxes):
        >>>     pass

        >>> # as a normal function
        >>> def converter_B(boxes):
        >>>     pass
        >>> register_box_converter('rbox', 'hbox', converter_B)
    """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method:
    # register_box_converter(src_type, dst_type, converter=Func)
    if converter is not None:
        _register_box_converter(
            src_type=src_type,
            dst_type=dst_type,
            converter=converter,
            force=force)
        return converter

    # use it as a decorator: @register_box_converter(name)
    def _register(func):
        _register_box_converter(
            src_type=src_type, dst_type=dst_type, converter=func, force=force)
        return func

    return _register

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

