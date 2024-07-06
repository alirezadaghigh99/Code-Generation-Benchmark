def rand(*size):
    if size == ():
        size = None
    return random_sample(size)

def randn(*size):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.randn(size, dtype=dtype)
    return array_or_scalar(values, return_scalar=size == ())

