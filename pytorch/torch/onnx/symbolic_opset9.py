def eye(g: jit_utils.GraphContext, *args):
    if len(args) == 5:
        # aten::eye(n, dtype, layout, device, pin_memory)
        n, dtype, layout, device, pin_memory = args
        dim_size = symbolic_helper._unsqueeze_helper(g, n, [0])
        shape = g.op("Concat", dim_size, dim_size, axis_i=0)
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)
    if len(args) == 6:
        # aten::eye(n, m, dtype, layout, device, pin_memory)
        n, m, dtype, layout, device, pin_memory = args
        shape = g.op(
            "Concat",
            symbolic_helper._unsqueeze_helper(g, n, [0]),
            symbolic_helper._unsqueeze_helper(g, m, [0]),
            axis_i=0,
        )
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)

    return symbolic_helper._unimplemented("aten::eye", f"with {len(args)} arguments")

def eye(g: jit_utils.GraphContext, *args):
    if len(args) == 5:
        # aten::eye(n, dtype, layout, device, pin_memory)
        n, dtype, layout, device, pin_memory = args
        dim_size = symbolic_helper._unsqueeze_helper(g, n, [0])
        shape = g.op("Concat", dim_size, dim_size, axis_i=0)
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)
    if len(args) == 6:
        # aten::eye(n, m, dtype, layout, device, pin_memory)
        n, m, dtype, layout, device, pin_memory = args
        shape = g.op(
            "Concat",
            symbolic_helper._unsqueeze_helper(g, n, [0]),
            symbolic_helper._unsqueeze_helper(g, m, [0]),
            axis_i=0,
        )
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)

    return symbolic_helper._unimplemented("aten::eye", f"with {len(args)} arguments")

def as_strided(g: jit_utils.GraphContext, self, sizes, strides, offset=None):
    sizes = symbolic_helper._maybe_get_const(sizes, "is")
    rank = len(strides)
    self_1d = symbolic_helper._reshape_helper(
        g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    )
    ind: Optional[torch.Tensor]
    if not symbolic_helper._is_value(sizes):
        ind = torch.tensor([0], dtype=torch.long)
        for i, (size, stride) in enumerate(zip(sizes, strides)):
            r_size = [1] * rank
            r_size[i] = -1
            ind = ind + torch.arange(size).view(r_size) * stride
        if offset:
            ind = ind + offset
        return g.op("Gather", self_1d, g.op("Constant", value_t=ind))
    else:
        ind = None
        for i, stride in enumerate(strides):
            r_size = [1] * rank
            r_size[i] = -1
            size = select(
                g,
                sizes,
                g.op("Constant", value_t=torch.tensor([0])),
                g.op("Constant", value_t=torch.tensor(i)),
            )
            tmp_ind = symbolic_helper._reshape_helper(
                g,
                arange(g, size, 4, None, None, None),
                g.op("Constant", value_t=torch.tensor(r_size)),
            )
            tmp_ind = g.op(
                "Mul", tmp_ind, g.op("Constant", value_t=torch.tensor([stride]))
            )
            if ind is None:
                ind = tmp_ind
            else:
                ind = g.op("Add", ind, tmp_ind)
        if offset:
            ind = g.op("Add", ind, g.op("Constant", torch.tensor([offset])))
        return g.op("Gather", self_1d, ind)

def lerp(g: jit_utils.GraphContext, self, end, weight):
    # Conditional for better numeric. This has been discussed in
    # https://github.com/pytorch/pytorch/pull/18871
    diff = g.op("Sub", end, self)
    return where(
        g,
        g.op("Less", weight, g.op("Constant", value_t=torch.tensor(0.5))),
        g.op("Add", self, g.op("Mul", weight, diff)),
        g.op(
            "Sub",
            end,
            g.op(
                "Mul",
                diff,
                g.op("Sub", g.op("Constant", value_t=torch.tensor(1.0)), weight),
            ),
        ),
    )

def linspace(
    g: jit_utils.GraphContext, start, end, steps, dtype, layout, device, pin_memory
):
    range_tensor = symbolic_helper._arange_helper(g, steps, None)
    step = div(
        g,
        sub(g, end, start),
        sub(g, steps, g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))),
    )
    return add(g, mul(g, range_tensor, step), start)

def multinomial(
    g: jit_utils.GraphContext, input, num_samples, replacement=False, generator=None
):
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Multinomial", "generator is not supported for multinomial", input
        )
    if not replacement and num_samples > 1:
        symbolic_helper._unimplemented(
            "Multinomial",
            "replacement=False when num_samples > 1 is not supported for multinomial",
            input,
        )

    log_input = log(g, input)
    return g.op(
        "Multinomial",
        log_input,
        dtype_i=_C_onnx.TensorProtoDataType.INT64,
        sample_size_i=num_samples,
    )

def isnan(g: jit_utils.GraphContext, input):
    output = g.op("IsNaN", input)
    return output

def linspace(
    g: jit_utils.GraphContext, start, end, steps, dtype, layout, device, pin_memory
):
    range_tensor = symbolic_helper._arange_helper(g, steps, None)
    step = div(
        g,
        sub(g, end, start),
        sub(g, steps, g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))),
    )
    return add(g, mul(g, range_tensor, step), start)

def convolution(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )

