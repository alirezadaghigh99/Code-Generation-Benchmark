def parse_model(model_text: str) -> onnx.ModelProto:
    """Parse a string to build a ModelProto.

    Arguments:
        model_text (string): formatted string
    Returns:
        ModelProto
    """
    (success, msg, model_proto_str) = C.parse_model(model_text)
    if success:
        return onnx.load_from_string(model_proto_str)
    raise ParseError(msg)