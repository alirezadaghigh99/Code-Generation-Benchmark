def class_mapping_not_available_in_environment(environment: dict) -> bool:
    return "CLASS_MAP" not in environment or not issubclass(
        type(environment["CLASS_MAP"]), dict
    )