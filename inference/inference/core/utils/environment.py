def str2bool(value: Any) -> bool:
    """
    Converts an environment variable to a boolean value.

    Args:
        value (str or bool): The environment variable value to be converted.

    Returns:
        bool: The converted boolean value.

    Raises:
        InvalidEnvironmentVariableError: If the value is not 'true', 'false', or a boolean.
    """
    if isinstance(value, bool):
        return value
    if not issubclass(type(value), str):
        raise InvalidEnvironmentVariableError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise InvalidEnvironmentVariableError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )