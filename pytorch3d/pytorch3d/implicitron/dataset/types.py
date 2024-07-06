def _dataclass_from_dict(d, typeannot):
    if d is None or typeannot is Any:
        return d
    is_optional, contained_type = _resolve_optional(typeannot)
    if is_optional:
        # an Optional not set to None, just use the contents of the Optional.
        return _dataclass_from_dict(d, contained_type)

    cls = get_origin(typeannot) or typeannot
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        types = cls.__annotations__.values()
        return cls(*[_dataclass_from_dict(v, tp) for v, tp in zip(d, types)])
    elif issubclass(cls, (list, tuple)):
        types = get_args(typeannot)
        if len(types) == 1:  # probably List; replicate for all items
            types = types * len(d)
        return cls(_dataclass_from_dict(v, tp) for v, tp in zip(d, types))
    elif issubclass(cls, dict):
        key_t, val_t = get_args(typeannot)
        return cls(
            (_dataclass_from_dict(k, key_t), _dataclass_from_dict(v, val_t))
            for k, v in d.items()
        )
    elif not dataclasses.is_dataclass(typeannot):
        return d

    assert dataclasses.is_dataclass(cls)
    fieldtypes = {f.name: _unwrap_type(f.type) for f in dataclasses.fields(typeannot)}
    return cls(**{k: _dataclass_from_dict(v, fieldtypes[k]) for k, v in d.items()})

