def _torch_inverse_cast(input: Tensor) -> Tensor:
    """Helper function to make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, Tensor):
        raise AssertionError(f"Input must be Tensor. Got: {type(input)}.")
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.linalg.inv(input.to(dtype)).to(input.dtype)

