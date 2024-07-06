def is_step_output_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return (
        selector_or_value.startswith("$steps.")
        and len(selector_or_value.split(".")) == 3
    )

def is_step_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return (
        selector_or_value.startswith("$steps.")
        and len(selector_or_value.split(".")) == 2
    )

def is_input_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return selector_or_value.startswith("$inputs")

def is_selector(selector_or_value: Any) -> bool:
    return str(selector_or_value).startswith("$")

