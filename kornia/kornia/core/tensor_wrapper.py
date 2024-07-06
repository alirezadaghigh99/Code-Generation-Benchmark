def wrap(v, cls):
    # wrap inputs if necessary
    if type(v) in {tuple, list}:
        return type(v)(wrap(vi, cls) for vi in v)

    return cls(v) if isinstance(v, Tensor) else v

