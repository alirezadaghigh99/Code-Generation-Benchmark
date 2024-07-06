def complex(real: TensorLikeType, imag: TensorLikeType) -> TensorLikeType:
    allowed_dtypes = (torch.float32, torch.float64, torch.float16)
    torch._check(
        real.dtype in allowed_dtypes and imag.dtype in allowed_dtypes,
        lambda: (
            f"Expected both inputs to be Half, Float or Double tensors but got "
            f"{real.dtype} and {imag.dtype}"
        ),
    )
    torch._check(
        real.dtype == imag.dtype,
        lambda: (
            f"Expected object of scalar type {real.dtype} but got "
            f"scalar type {imag.dtype} for second argument"
        ),
    )
    result_dtype = utils.corresponding_complex_dtype(real.dtype)  # type: ignore[arg-type]
    common_shape = _broadcast_shapes(real.shape, imag.shape)
    result = real.new_empty(
        common_shape,
        dtype=result_dtype,
        layout=real.layout,
        device=real.device,
        # pin_memory=real.is_pinned(),  # NYI
    )
    result.real = real
    result.imag = imag
    return result

