def serialise_workflow_result(
    result: Dict[str, Any],
    excluded_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if excluded_fields is None:
        excluded_fields = []
    excluded_fields = set(excluded_fields)
    serialised_result = {}
    for key, value in result.items():
        if key in excluded_fields:
            continue
        if contains_image(element=value):
            value = serialise_image(image=value)
        elif issubclass(type(value), dict):
            value = serialise_dict(elements=value)
        elif issubclass(type(value), list):
            value = serialise_list(elements=value)
        serialised_result[key] = value
    return serialised_result