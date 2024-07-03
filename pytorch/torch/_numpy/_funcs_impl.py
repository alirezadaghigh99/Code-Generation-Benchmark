def logspace(
    start,
    stop,
    num=50,
    endpoint=True,
    base=10.0,
    dtype: Optional[DTypeLike] = None,
    axis=0,
):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    return torch.logspace(start, stop, num, base=base, dtype=dtype)