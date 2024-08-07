def get_all_math_dtypes(device) -> List[torch.dtype]:
    return (
        get_all_int_dtypes()
        + get_all_fp_dtypes(
            include_half=device.startswith("cuda"), include_bfloat16=False
        )
        + get_all_complex_dtypes()
    )

def all_types_and_complex_and(*dtypes):
    return _all_types_and_complex + _validate_dtypes(*dtypes)

def integral_types():
    return _integral_types

