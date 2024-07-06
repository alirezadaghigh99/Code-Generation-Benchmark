def _create_model():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
    pads = helper.make_tensor_value_info("pads", TensorProto.FLOAT, [1, 4])
    value = helper.make_tensor_value_info("value", AttributeProto.FLOAT, [1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])
    node_def = helper.make_node(
        "Pad",
        ["X", "pads", "value"],
        ["Y"],
        mode="constant",
    )
    graph_def = helper.make_graph(
        [node_def],
        "test-model",
        [X, pads, value],
        [Y],
    )
    return helper.make_model(graph_def, producer_name="onnx-example")

