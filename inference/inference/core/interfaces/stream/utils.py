def wrap_in_list(element: Union[T, List[T]]) -> List[T]:
    if not issubclass(type(element), list):
        element = [element]
    return element