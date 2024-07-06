def convert_keys_to_snake_case(d):
    """Convert all keys in a dictionary from camelCase to snake_case.

    Args:
        d: the dictionary

    Returns:
        a dictionary with snake case keys
    """
    if isinstance(d, dict):
        return {
            (
                camel_to_snake(k) if k != FO3D_VERSION_KEY else k
            ): convert_keys_to_snake_case(v)
            for k, v in d.items()
        }
    elif isinstance(d, list):
        return [convert_keys_to_snake_case(item) for item in d]
    else:
        return d

