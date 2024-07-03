def get_args_of(base_cls: type, cls, error_msg: str) -> tuple[TypeLike, ...]:
    """Equivalent to `get_args(cls)`, except that it tracks through the type hierarchy
    finding the way in which `cls` subclasses `base_cls`, and returns the arguments that
    subscript that instead.

    For example,
    ```python
    class Foo(Generic[T]):
        pass

    class Bar(Generic[S]):
        pass

    class Qux(Foo[T], Bar[S]):
        pass

    get_args_of(Foo, Qux[int, str], "...")  # int
    ```

    In addition, any unfilled type variables are returned as `Any`.

    **Arguments:**

    - `base_cls`: the class to get parameters with respect to.
    - `cls`: the class or subscripted generic to get arguments with respect to.
    - `error_msg`: if anything goes wrong, mention this in the error message.

    **Returns:**

    A tuple of types indicating the arguments. In addition, any unfilled type variables
    are returned as `Any`.
    """

    if not inspect.isclass(base_cls):
        raise TypeError(f"{base_cls} should be a class")
    if not hasattr(base_cls, "__parameters__"):
        raise TypeError(f"{base_cls} should be an unsubscripted generic")

    origin = get_origin_no_specials(cls, error_msg)
    if inspect.isclass(cls):
        # Unsubscripted
        assert origin is None
        origin = cls
        params = [Any for _ in getattr(cls, "__parameters__", ())]
    else:
        # Subscripted
        assert origin is not None
        params: list[TypeLike] = []
        for param in get_args(cls):
            if isinstance(param, TypeVar):
                params.append(Any)
            else:
                params.append(param)
    if issubclass(origin, base_cls):
        out = _get_args_of_impl(base_cls, origin, tuple(params), error_msg)
        if out is None:
            # Dependency is purely inheritance without subscripting
            return tuple(Any for _ in base_cls.__parameters__)
        else:
            return out
    else:
        raise TypeError(f"{cls} is not a subclass of {base_cls}")