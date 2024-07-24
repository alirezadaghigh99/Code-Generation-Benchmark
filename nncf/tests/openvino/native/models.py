class LinearModel(OVReferenceModel):
    def _create_ov_model(self, input_shape=None, reshape_shape=None, matmul_w_shape=None, add_shape=None):
        if input_shape is None:
            input_shape = [1, 3, 4, 2]
        if reshape_shape is None:
            reshape_shape = (1, 3, 2, 4)
        if matmul_w_shape is None:
            matmul_w_shape = (4, 5)
        if add_shape is None:
            add_shape = (1, 3, 2, 4)

        input_1 = opset.parameter(input_shape, name="Input")
        reshape = opset.reshape(input_1, reshape_shape, special_zero=False, name="Reshape")
        data = self._rng.random(matmul_w_shape).astype(np.float32) - 0.5
        matmul = opset.matmul(reshape, data, transpose_a=False, transpose_b=False, name="MatMul")
        add = opset.add(reshape, self._rng.random(add_shape).astype(np.float32), name="Add")
        r1 = opset.result(matmul, name="Result_MatMul")
        # TODO(KodiaqQ): Remove this after fix - CVS-100010
        r1.get_output_tensor(0).set_names(set(["Result_MatMul"]))
        r2 = opset.result(add, name="Result_Add")
        r2.get_output_tensor(0).set_names(set(["Result_Add"]))
        model = ov.Model([r1, r2], [input_1])
        return model

class FPModel(OVReferenceModel):
    def __init__(self, const_dtype="FP32", input_dtype="FP32"):
        self.const_dtype = np.float32 if const_dtype == "FP32" else np.float16
        self.input_dtype = np.float32 if input_dtype == "FP32" else np.float16
        super().__init__()

    def _create_ov_model(self):
        input_shape = [1, 3, 4, 2]
        input_1 = opset.parameter(input_shape, name="Input", dtype=self.input_dtype)
        data = self._rng.random((1, 3, 4, 5)).astype(self.const_dtype)
        if self.const_dtype != self.input_dtype:
            data = opset.convert(data, self.input_dtype)
        matmul = opset.matmul(input_1, data, transpose_a=True, transpose_b=False, name="MatMul")
        bias = self._rng.random((1, 3, 1, 1)).astype(self.const_dtype)
        if self.const_dtype != self.input_dtype:
            bias = opset.convert(bias, self.input_dtype)
        add = opset.add(matmul, bias, name="Add")
        result = opset.result(add, name="Result_Add")
        result.get_output_tensor(0).set_names(set(["Result_Add"]))
        model = ov.Model([result], [input_1])
        return model

class QuantizedModel(OVReferenceModel):
    @staticmethod
    def _create_fq_node(parent_node, name):
        # OV bug with FQ element types after fusing preprocessing
        return opset.fake_quantize(
            parent_node, np.float32(-1), np.float32(1), np.float32(-1), np.float32(1), 256, name=name
        )

    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 14, 28], name="Input_1")
        conv_1_fq_input = self._create_fq_node(input_1, name="Conv_1/fq_input_0")

        mean = self._rng.random((1, 3, 1, 1)).astype(np.float32)
        scale = self._rng.random((1, 3, 1, 1)).astype(np.float32) + 1e-4
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32) / scale
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv_1_fq_weights = self._create_fq_node(kernel, name="Conv_1/fq_weights_0")
        conv_1 = opset.convolution(conv_1_fq_input, conv_1_fq_weights, strides, pads, pads, dilations, name="Conv_1")
        relu_1 = opset.relu(conv_1, name="Relu_1")

        input_2 = opset.parameter([1, 3, 28, 14], name="Input_2")
        multiply = opset.multiply(input_2, 1 / scale, name="Mul")
        add_1 = opset.add(multiply, (-1) * mean, name="Add_1")
        transpose_fq_input = self._create_fq_node(add_1, name="Transpose/fq_input_0")
        transpose = opset.transpose(transpose_fq_input, [0, 1, 3, 2], name="Transpose")

        cat_fq_input = self._create_fq_node(relu_1, name="Concat_1/fq_input_0")
        cat_1 = opset.concat([cat_fq_input, transpose], axis=1, name="Concat_1")

        kernel = self._rng.random((12, 6, 1, 1)).astype(np.float32)
        conv_2_fq_weights = self._create_fq_node(kernel, name="Conv_2/fq_weights_0")
        conv_2 = opset.convolution(cat_1, conv_2_fq_weights, strides, pads, pads, dilations, name="Conv_2")
        relu_2 = opset.relu(conv_2, name="Relu_2")

        kernel = self._rng.random((6, 12, 1, 1)).astype(np.float32)
        conv_3_fq_input = self._create_fq_node(relu_2, name="Conv_3/fq_input_0")
        conv_3_fq_weights = self._create_fq_node(kernel, name="Conv_3/fq_weights_0")
        conv_3 = opset.convolution(conv_3_fq_input, conv_3_fq_weights, strides, pads, pads, dilations, name="Conv_3")

        mean = self._rng.random((1, 6, 1, 1)).astype(np.float32)
        add_2_const = opset.constant((-1) * mean)
        add_2_fq_weights = self._create_fq_node(add_2_const, name="Add_2/fq_weights_0")
        add_2 = opset.add(cat_1, add_2_fq_weights, name="Add_2")

        cat_2 = opset.concat([conv_3, add_2], axis=1, name="Concat_2")

        reshape = opset.reshape(cat_2, (-1, 2352), True)
        matmul_constant = self._rng.random((100, 2352)).astype(np.float32)
        matmul = opset.matmul(reshape, matmul_constant, False, True)
        result = opset.result(matmul, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1, input_2])
        return model

class SplitConcatModel(OVReferenceModel):
    def _create_ov_model(self, input_name) -> ov.Model:
        input_1 = opset.parameter([1, 3, 3, 3], name=input_name)
        split = opset.split(input_1, 1, 3, name="split")
        add_const = np.array(1).astype(np.float32)
        add_1 = opset.add(split.output(0), add_const, name="add_1")
        add_2 = opset.add(split.output(1), add_const, name="add_2")
        add_3 = opset.add(split.output(2), add_const, name="add_3")
        concat = opset.concat([add_1, add_2, add_3], 1, name="concat")
        add_4 = opset.add(concat, add_const, name="add_4")
        add_5 = opset.add(concat, add_const, name="add_5")
        result_1 = opset.result(add_4, name="result_1")
        result_2 = opset.result(add_5, name="result_2")
        model = ov.Model([result_1, result_2], [input_1])
        return model

class SharedConvModel(OVReferenceModel):
    def _create_ov_model(self, input_name="Input", input_shape=(1, 3, 3, 3), kernel=None) -> ov.Model:
        input_1 = opset.parameter(input_shape, name=input_name)
        if kernel is None:
            c_in = input_shape[1]
            kernel = self._rng.random((3, c_in, 1, 1))
        const_kernel = opset.constant(kernel, np.float32, name="Shared_conv_w")
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv_1 = opset.convolution(input_1, const_kernel, strides, pads, pads, dilations, name="Conv_1")
        conv_2 = opset.convolution(input_1, const_kernel, strides, pads, pads, dilations, name="Conv_2")
        result_1 = opset.result(conv_1, name="Result_1")
        result_2 = opset.result(conv_2, name="Result_2")
        model = ov.Model([result_1, result_2], [input_1])
        return model

class SimpleSplitModel(OVReferenceModel):
    def _create_ov_model(self, input_shape=None, splits=None):
        if input_shape is None:
            input_shape = [1, 9, 4, 4]
            splits = 3
        input_1 = opset.parameter(input_shape, name="Input")
        split = opset.split(input_1, 1, splits, name="Split")
        results = []
        for idx, output in enumerate(split.outputs()):
            results.append(opset.result(output, name=f"Result_{idx}"))

        model = ov.Model(results, [input_1])
        return model

class IntegerModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 7, 1], name="Input")
        convert_1 = opset.convert(input_1, destination_type="i64", name="Convert_1")

        gather_1 = opset.gather(convert_1, 2, axis=0, batch_dims=0)
        gather_1.set_friendly_name("Gather_1")

        gather_2_data = opset.constant(self._rng.random((3, 6)), dtype=np.float32, name="gather_2_data")
        gather_2 = opset.gather(gather_2_data, gather_1, axis=0, batch_dims=0)
        gather_2.set_friendly_name("Gather_2")

        gather_3 = opset.gather(gather_2, 2, axis=0, batch_dims=0)
        gather_3.set_friendly_name("Gather_3")

        matmul_1_data = opset.constant(self._rng.random((6, 6)), dtype=np.float32, name="matmul_1_data")
        matmul_1 = opset.matmul(gather_3, matmul_1_data, transpose_a=False, transpose_b=True, name="MatMul_1")

        gather_4 = opset.gather(input_1, 0, axis=2, batch_dims=0)
        gather_4.set_friendly_name("Gather_4")

        matmul_2_data = opset.constant(self._rng.random((6, 7)), dtype=np.float32, name="matmul_2_data")
        matmul_2 = opset.matmul(gather_4, matmul_2_data, transpose_a=False, transpose_b=True, name="MatMul_2")
        add_1 = opset.add(matmul_1, matmul_2, name="Add_1")

        result = opset.result(add_1, name="Result")
        model = ov.Model([result], [input_1])
        return model

class WeightsModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 5, 5], name="Input_1")
        kernel_data = self._rng.random((3, 3, 1, 1)).astype(np.float32)
        kernel = opset.constant(kernel_data, dtype=np.float32, name="conv_weights_0")
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.convolution(input_1, kernel, strides, pads, pads, dilations, name="Conv")
        kernel_data_2 = self._rng.random((3, 3, 1, 1)).astype(np.float32)
        kernel_2 = opset.constant(kernel_data_2, dtype=np.float32, name="conv_weights_1")
        output_shape = [1, 1]
        conv_tr = opset.convolution_backprop_data(
            conv, kernel_2, output_shape, strides, pads, pads, dilations, name="Conv_backprop"
        )

        weights_1 = opset.constant(self._rng.random((1, 4)), dtype=np.float32, name="weights_1")
        matmul_1 = opset.matmul(conv_tr, weights_1, transpose_a=False, transpose_b=False, name="MatMul_1")
        weights_0 = opset.constant(self._rng.random((1, 1)), dtype=np.float32, name="weights_0")
        matmul_0 = opset.matmul(weights_0, matmul_1, transpose_a=False, transpose_b=False, name="MatMul_0")
        matmul = opset.matmul(matmul_0, matmul_1, transpose_a=False, transpose_b=True, name="MatMul")
        matmul_const = opset.matmul(weights_1, weights_0, transpose_a=True, transpose_b=False, name="MatMul_const")

        add = opset.add(matmul_const, matmul)
        result = opset.result(add, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model

class AWQMatmulModel(OVReferenceModel):
    """
    Model for testing AWQ algorithm. Contains MatMul->Multiply->MatMul pattern.
    """

    @staticmethod
    def get_weights(weights_data, is_int8, name):
        if not is_int8:
            return opset.constant(weights_data, dtype=np.float32, name=name)
        else:
            qw = opset.constant(weights_data, dtype=np.uint8, name="qw_" + name)
            qw = opset.convert(qw, destination_type=np.float32)

            zp = opset.constant(np.array([2**7]), dtype=np.uint8, name="zp_" + name)
            zp = opset.convert(zp, destination_type=np.float32)

            scale = opset.constant(
                np.ones((weights_data.shape[0], 1), dtype=np.float32), dtype=np.float32, name="scale_" + name
            )
            return (qw - zp) * scale

    def _create_ov_model(self, is_int8=False):
        input_node = opset.parameter([8, 8], name="Input_1")

        weights_data1 = np.arange(0, 64).reshape(8, 8)
        weights_data1[:] = 2.0
        weights1 = self.get_weights(weights_data1, is_int8, name="weights_1")
        node1 = opset.matmul(input_node, weights1, transpose_a=False, transpose_b=True, name="MatMul_1")

        weights_data2 = np.arange(0, 64).reshape(8, 8)
        weights_data2[:] = 3.0
        weights2 = self.get_weights(weights_data2, is_int8, name="weights_2")
        node2 = opset.matmul(input_node, weights2, transpose_a=False, transpose_b=True, name="MatMul_2")

        node_multiply = opset.multiply(node1, node2, name="Multiply")

        weights_data3 = np.arange(0, 64).reshape(8, 8)
        weights_data3[:] = 4.0
        weights3 = self.get_weights(weights_data3, is_int8, name="weights_3")
        node3 = opset.matmul(node_multiply, weights3, transpose_a=False, transpose_b=True, name="MatMul_3")

        weights_data4 = np.arange(0, 64).reshape(8, 8)
        weights_data4[:] = 2.0
        weights4 = self.get_weights(weights_data4, is_int8, name="weights_4")
        node4 = opset.matmul(node3, weights4, transpose_a=False, transpose_b=True, name="MatMul_4")

        weights_data5 = np.arange(0, 64).reshape(8, 8)
        weights_data5[:] = 3.0
        weights5 = self.get_weights(weights_data5, is_int8, name="weights_5")
        node5 = opset.matmul(node3, weights5, transpose_a=False, transpose_b=True, name="MatMul_5")

        node_multiply_2 = opset.multiply(node4, node5, name="Multiply_2")

        weights_data6 = np.arange(0, 64).reshape(8, 8)
        weights_data6[:] = 4.0
        weights6 = self.get_weights(weights_data6, is_int8, name="weights_6")
        node6 = opset.matmul(node_multiply_2, weights6, transpose_a=False, transpose_b=True, name="MatMul_6")

        result = opset.result(node6, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_node])
        return model

