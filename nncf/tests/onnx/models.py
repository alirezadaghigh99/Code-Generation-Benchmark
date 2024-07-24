class IdentityConvolutionalModel(ONNXReferenceModel):
    def __init__(self, input_shape=None, inp_ch=3, out_ch=32, kernel_size=1, conv_w=None, conv_b=None):
        if input_shape is None:
            input_shape = [1, 3, 10, 10]

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)

        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = inp_ch, out_ch, (kernel_size,) * 2
        rng = get_random_generator()
        conv1_W = conv_w
        if conv1_W is None:
            conv1_W = rng.uniform(0, 1, (conv1_out_channels, conv1_in_channels, *conv1_kernel_shape))
        conv1_W = conv1_W.astype(np.float32)

        conv1_B = conv_b
        if conv1_B is None:
            conv1_B = rng.uniform(0, 1, conv1_out_channels)
        conv1_B = conv1_B.astype(np.float32)

        model_identity_op_name = "Identity"
        model_conv_op_name = "Conv1"
        model_output_name = "Y"

        identity_node = onnx.helper.make_node(
            name=model_identity_op_name,
            op_type="Identity",
            inputs=[model_input_name],
            outputs=[model_input_name + "_X"],
        )

        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )

        conv1_node = onnx.helper.make_node(
            name=model_conv_op_name,
            op_type="Conv",
            inputs=[model_input_name + "_X", conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
        )

        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [1, conv1_out_channels, input_shape[-2] - kernel_size + 1, input_shape[-1] - kernel_size + 1],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[identity_node, conv1_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "one_convolutional_model.dot")

