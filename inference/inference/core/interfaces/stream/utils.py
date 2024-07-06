def broadcast_elements(
    elements: List[T],
    desired_length: int,
    error_description: str,
) -> List[T]:
    if len(elements) == desired_length:
        return elements
    if len(elements) != 1:
        raise ValueError(error_description)
    return elements * desired_length

def wrap_in_list(element: Union[T, List[T]]) -> List[T]:
    if not issubclass(type(element), list):
        element = [element]
    return element

