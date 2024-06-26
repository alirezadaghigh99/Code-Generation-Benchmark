def is_selector(selector_or_value: Any) -> bool:
    return str(selector_or_value).startswith("$")