def convert_precision_str_to_dtype(precision: str) -> Optional[torch.dtype]:
    """
    Converts precision as a string to a torch.dtype

    Args:
        precision: string containing the precision

    Raises:
        ValueError if an invalid precision string is passed.

    """
    if precision not in _DTYPE_STRING_TO_DTYPE_MAPPING:
        raise ValueError(
            f"Precision {precision} not supported. Please use one of {list(_DTYPE_STRING_TO_DTYPE_MAPPING.keys())}"
        )
    return _DTYPE_STRING_TO_DTYPE_MAPPING[precision]