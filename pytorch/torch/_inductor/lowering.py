def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    assert isinstance(size, (list, tuple))
    assert isinstance(stride, (list, tuple, type(None)))
    assert_nyi(not pin_memory, "pin_memory")
    assert_nyi(layout in (None, torch.strided), f"layout={layout}")
    dtype = decode_dtype(dtype) or torch.get_default_dtype()
    device = device or torch.tensor(0.0).device
    pointwise = _full(fill_value=0, device=device, dtype=dtype, size=size)
    pointwise.realize()
    buffer = pointwise.data.data
    # explicitly set ranges to zeros in order to make a NopKernelSchedulerNode
    buffer.data.ranges = [0] * len(size)
    assert isinstance(buffer, ir.ComputedBuffer)
    size = [sympy.expand(s) for s in size]
    stride = (
        [sympy.expand(s) for s in stride]
        if stride
        else ir.FlexibleLayout.contiguous_strides(size)
    )
    buffer.layout = ir.FixedLayout(
        device=device,
        dtype=dtype,
        size=size,
        stride=stride,
    )
    return pointwise