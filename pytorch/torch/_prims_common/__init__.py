def corresponding_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return _real_to_complex_dtype_map[dtype]

def compare_tensor_meta(
    a: TensorLikeType,
    b: TensorLikeType,
    check_strides=False,
    *,
    allow_rhs_unbacked=False,
    check_conj=True,
):
    """
    Checks that two tensor likes have the same shape,
    dtype and device.

    In the future this will validate additional metadata, like
    strides.
    """
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)

    if not same_shape(a.shape, b.shape, allow_rhs_unbacked=allow_rhs_unbacked):
        msg = f"Shapes {a.shape} and {b.shape} are not equal!"
        raise AssertionError(msg)

    if a.dtype != b.dtype:
        msg = f"Dtypes {a.dtype} and {b.dtype} are not equal!"
        raise AssertionError(msg)

    if a.device != b.device:
        # Handles special cuda:0 vs cuda case
        # TODO: we should review why this happens and see about fixing it
        if (str(a.device) == "cuda:0" or str(a.device) == "cuda") and (
            str(b.device) == "cuda:0" or str(b.device) == "cuda"
        ):
            pass
        else:
            msg = f"Devices {a.device} and {b.device} are not equal!"
            raise AssertionError(msg)

    # Stride checking is currently disabled, see https://github.com/pytorch/pytorch/issues/78050
    if check_strides:
        same_strides, idx = check_significant_strides(a, b)
        if not same_strides:
            msg = f"Stride mismatch! Strides are {a.stride()} and {b.stride()} (mismatched at {idx})!"
            raise RuntimeError(msg)

        if a.storage_offset() != b.storage_offset():
            msg = f"Storage offset mismatch! Storage offsets are {a.storage_offset()} and {b.storage_offset()}!"
            raise RuntimeError(msg)

    if check_conj:
        if a.is_conj() != b.is_conj():
            raise RuntimeError(
                f"Conj mismatch! is_conj is set to {a.is_conj()} and {b.is_conj()}"
            )

    if a.is_neg() != b.is_neg():
        raise RuntimeError(
            f"Neg mismatch! is_neg is set to {a.is_neg()} and {b.is_neg()}"
        )

