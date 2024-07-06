def as_list(val):
    """Ensure the input value is converted into a list.

    This helper function takes an input value and ensures that it is always returned as a list.

    - If the input is a single value, it will be wrapped in a list.
    - If the input is an iterable, it will be converted into a list.

    Parameters
    ----------
    val : object
        The input value that needs to be converted into a list.

    Returns
    -------
    list
        The input value as a list.

    Examples
    --------
    ```py
    as_list("test")
    # ['test']

    as_list(["test1", "test2"])
    # ['test1', 'test2']
    ```
    """
    treat_single_value = str

    if isinstance(val, treat_single_value):
        return [val]

    if hasattr(val, "__iter__"):
        return list(val)

    return [val]

