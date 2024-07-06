def to_text(proto: onnx.ModelProto | onnx.FunctionProto | onnx.GraphProto) -> str:
    if isinstance(proto, onnx.ModelProto):
        return C.model_to_text(proto.SerializeToString())
    if isinstance(proto, onnx.FunctionProto):
        return C.function_to_text(proto.SerializeToString())
    if isinstance(proto, onnx.GraphProto):
        return C.graph_to_text(proto.SerializeToString())
    raise TypeError("Unsupported argument type.")

