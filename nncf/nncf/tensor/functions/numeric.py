def allclose(
    a: Tensor, b: Union[Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    :param a: The first input tensor.
    :param b: The second input tensor.
    :param rtol: The relative tolerance parameter, defaults to 1e-05.
    :param atol: The absolute tolerance parameter, defaults to 1e-08.
    :param equal_nan: Whether to compare NaN`s as equal. If True,
      NaN`s in a will be considered equal to NaN`s in b in the output array.
      Defaults to False.
    :return: True if the two arrays are equal within the given tolerance, otherwise False.
    """
    return allclose(a.data, unwrap_tensor_data(b), rtol=rtol, atol=atol, equal_nan=equal_nan)

