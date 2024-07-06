def quantize_per_tensor(g: jit_utils.GraphContext, input, scale, zero_point, dtype):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # TODO(justinchuby): Extract all the cast ops into a helper function.
    zero_point = g.op(
        "Cast", zero_point, to_i=_type_utils.JitScalarType(dtype).onnx_type()
    )
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return symbolic_helper.quantize_helper(g, input, scale, zero_point)

