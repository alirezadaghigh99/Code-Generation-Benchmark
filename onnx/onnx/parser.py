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

def parse_graph(graph_text: str) -> onnx.GraphProto:
    """Parse a string to build a GraphProto.

    Arguments:
        graph_text (string): formatted string
    Returns:
        GraphProto
    """
    (success, msg, graph_proto_str) = C.parse_graph(graph_text)
    if success:
        graph_proto = onnx.GraphProto()
        graph_proto.ParseFromString(graph_proto_str)
        return graph_proto
    raise ParseError(msg)

