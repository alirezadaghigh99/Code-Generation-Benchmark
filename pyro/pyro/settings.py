def get(alias: Optional[str] = None) -> Any:
    """
    Gets one or all global settings.

    :param str alias: The name of a registered setting.
    :returns: The currently set value.
    """
    if alias is None:
        # Return dict of all settings.
        return {alias: get(alias) for alias in sorted(_REGISTRY)}
    # Get a single setting.
    module, deepname, validator = _REGISTRY[alias]
    value = import_module(module)
    for name in deepname.split("."):
        value = getattr(value, name)
    return value

