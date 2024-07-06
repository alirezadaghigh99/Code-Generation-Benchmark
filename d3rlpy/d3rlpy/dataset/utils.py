def check_dtype(array: Union[NDArray, Sequence[NDArray]], dtype: DType) -> bool:
    if isinstance(array, (list, tuple)):
        return all(v.dtype == dtype for v in array)
    elif isinstance(array, np.ndarray):
        return array.dtype == dtype
    else:
        raise ValueError(f"invalid array type: {type(array)}")

