def get_default_args(C, *, _do_not_process: Tuple[type, ...] = ()) -> DictConfig:
    """
    Get the DictConfig corresponding to the defaults in a dataclass or
    configurable. Normal use is to provide a dataclass can be provided as C.
    If enable_get_default_args has been called on a function or plain class,
    then that function or class can be provided as C.

    If C is a subclass of Configurable or ReplaceableBase, we make sure
    it has been processed with expand_args_fields.

    Args:
        C: the class or function to be processed
        _do_not_process: (internal use) When this function is called from
                    expand_args_fields, we specify any class currently being
                    processed, to make sure we don't try to process a class
                    while it is already being processed.

    Returns:
        new DictConfig object, which is typed.
    """
    if C is None:
        return DictConfig({})

    if _is_configurable_class(C):
        if C in _do_not_process:
            raise ValueError(
                f"Internal recursion error. Need processed {C},"
                f" but cannot get it. _do_not_process={_do_not_process}"
            )
        # This is safe to run multiple times. It will return
        # straight away if C has already been processed.
        expand_args_fields(C, _do_not_process=_do_not_process)

    if dataclasses.is_dataclass(C):
        # Note that if get_default_args_field is used somewhere in C,
        # this call is recursive. No special care is needed,
        # because in practice get_default_args_field is used for
        # separate types than the outer type.

        try:
            out: DictConfig = OmegaConf.structured(C)
        except Exception:
            print(f"### OmegaConf.structured({C}) failed ###")
            # We don't use `raise From` here, because that gets the original
            # exception hidden by the OC_CAUSE logic in the case where we are
            # called by hydra.
            raise
        exclude = getattr(C, "_processed_members", ())
        with open_dict(out):
            for field in exclude:
                out.pop(field, None)
        return out

    if _is_configurable_class(C):
        raise ValueError(f"Failed to process {C}")

    if not inspect.isfunction(C) and not inspect.isclass(C):
        raise ValueError(f"Unexpected {C}")

    dataclass_name = _dataclass_name_for_function(C)
    dataclass = getattr(sys.modules[C.__module__], dataclass_name, None)
    if dataclass is None:
        raise ValueError(
            f"Cannot get args for {C}. Was enable_get_default_args forgotten?"
        )

    try:
        out: DictConfig = OmegaConf.structured(dataclass)
    except Exception:
        print(f"### OmegaConf.structured failed for {C.__name__} ###")
        raise
    return outdef _is_actually_dataclass(some_class) -> bool:
    # Return whether the class some_class has been processed with
    # the dataclass annotation. This is more specific than
    # dataclasses.is_dataclass which returns True on anything
    # deriving from a dataclass.

    # Checking for __init__ would also work for our purpose.
    return "__dataclass_fields__" in some_class.__dict__def get_default_args(C, *, _do_not_process: Tuple[type, ...] = ()) -> DictConfig:
    """
    Get the DictConfig corresponding to the defaults in a dataclass or
    configurable. Normal use is to provide a dataclass can be provided as C.
    If enable_get_default_args has been called on a function or plain class,
    then that function or class can be provided as C.

    If C is a subclass of Configurable or ReplaceableBase, we make sure
    it has been processed with expand_args_fields.

    Args:
        C: the class or function to be processed
        _do_not_process: (internal use) When this function is called from
                    expand_args_fields, we specify any class currently being
                    processed, to make sure we don't try to process a class
                    while it is already being processed.

    Returns:
        new DictConfig object, which is typed.
    """
    if C is None:
        return DictConfig({})

    if _is_configurable_class(C):
        if C in _do_not_process:
            raise ValueError(
                f"Internal recursion error. Need processed {C},"
                f" but cannot get it. _do_not_process={_do_not_process}"
            )
        # This is safe to run multiple times. It will return
        # straight away if C has already been processed.
        expand_args_fields(C, _do_not_process=_do_not_process)

    if dataclasses.is_dataclass(C):
        # Note that if get_default_args_field is used somewhere in C,
        # this call is recursive. No special care is needed,
        # because in practice get_default_args_field is used for
        # separate types than the outer type.

        try:
            out: DictConfig = OmegaConf.structured(C)
        except Exception:
            print(f"### OmegaConf.structured({C}) failed ###")
            # We don't use `raise From` here, because that gets the original
            # exception hidden by the OC_CAUSE logic in the case where we are
            # called by hydra.
            raise
        exclude = getattr(C, "_processed_members", ())
        with open_dict(out):
            for field in exclude:
                out.pop(field, None)
        return out

    if _is_configurable_class(C):
        raise ValueError(f"Failed to process {C}")

    if not inspect.isfunction(C) and not inspect.isclass(C):
        raise ValueError(f"Unexpected {C}")

    dataclass_name = _dataclass_name_for_function(C)
    dataclass = getattr(sys.modules[C.__module__], dataclass_name, None)
    if dataclass is None:
        raise ValueError(
            f"Cannot get args for {C}. Was enable_get_default_args forgotten?"
        )

    try:
        out: DictConfig = OmegaConf.structured(dataclass)
    except Exception:
        print(f"### OmegaConf.structured failed for {C.__name__} ###")
        raise
    return out