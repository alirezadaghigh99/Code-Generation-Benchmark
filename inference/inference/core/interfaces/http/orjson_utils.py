def contains_image(element: Any) -> bool:
    return (
        issubclass(type(element), dict)
        and element.get("type") == ImageType.NUMPY_OBJECT.value
    )