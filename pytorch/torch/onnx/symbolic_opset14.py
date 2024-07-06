def scaled_dot_product_attention(
    g: jit_utils.GraphContext,
    query: torch._C.Value,
    key: torch._C.Value,
    value: torch._C.Value,
    attn_mask: Optional[torch._C.Value] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[torch._C.Value] = None,
):
    assert (not is_causal) or (
        is_causal and symbolic_helper._is_none(attn_mask)
    ), "is_causal and attn_mask cannot be set at the same time"

    scale = symbolic_helper._maybe_get_const(scale, "f")
    if symbolic_helper._is_none(scale):
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # Swap the last two axes of key
    # NOTE: onnx-script has different logic here, because the attribute perms in
    # transpose needs list of ints
    key_shape_builtin = symbolic_helper._get_tensor_rank(key)
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

    if symbolic_helper._is_none(attn_mask):
        mul_qk_add = mul_qk
    elif (
        _type_utils.JitScalarType.from_value(attn_mask)
        == _type_utils.JitScalarType.BOOL
    ):
        # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    elif _type_utils.JitScalarType.from_value(attn_mask) in (
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    ):
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    else:
        raise ValueError(
            f"Unsupported type for attn_mask: {_type_utils.JitScalarType.from_value(attn_mask)}"
        )

    attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)

    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )

    return g.op("MatMul", attn_weight, value)

