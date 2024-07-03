def _slice_argnums(args, argnums, as_tuple=True):
    if not isinstance(argnums, int) and not isinstance(argnums, tuple):
        raise RuntimeError(
            f"argnums must be int or Tuple[int, ...], got: {type(argnums)}"
        )
    argnums = _validate_and_wrap_argnums(argnums, len(args))
    _check_unique_non_empty(argnums)
    if isinstance(argnums, int):
        if as_tuple:
            return (args[argnums],)
        else:
            return args[argnums]
    return tuple(args[i] for i in argnums)