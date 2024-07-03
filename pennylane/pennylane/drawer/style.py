def _set_style(style: str = None):
    """
    Execute a style function to change the current rcParams.

    Args:
        style (Optional[str]): A style specification. If no style is provided,
            the latest style set with ``use_style`` is used instead.
    """
    if not style:
        __current_style_fn()
    elif style in _styles_map:
        _styles_map[style]()
    else:
        raise TypeError(
            f"style '{style}' provided to ``_set_style`` "
            f"does not exist.  Available options are {available_styles()}"
        )