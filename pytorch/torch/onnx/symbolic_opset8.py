def empty(
    g: jit_utils.GraphContext,
    sizes,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    return zeros(g, sizes, dtype, layout, device, pin_memory)def empty_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    return zeros_like(g, input, dtype, layout, device, pin_memory)def empty(
    g: jit_utils.GraphContext,
    sizes,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    return zeros(g, sizes, dtype, layout, device, pin_memory)