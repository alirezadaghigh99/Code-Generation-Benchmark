class VerificationOptions:
    """Options for ONNX export verification.

    Attributes:
        flatten: If True, unpack nested list/tuple/dict inputs into a flattened list of
            Tensors for ONNX. Set this to False if nested structures are to be preserved
            for ONNX, which is usually the case with exporting ScriptModules. Default True.
        ignore_none: Whether to ignore None type in torch output, which is usually the
            case with tracing. Set this to False, if torch output should keep None type,
            which is usually the case with exporting ScriptModules. Default to True.
        check_shape: Whether to check the shapes between PyTorch and ONNX Runtime outputs
            are exactly the same. Set this to False to allow output shape broadcasting.
            Default to True.
        check_dtype: Whether to check the dtypes between PyTorch and ONNX Runtime outputs
            are consistent. Default to True.
        backend: ONNX backend for verification. Default to OnnxBackend.ONNX_RUNTIME_CPU.
        rtol: relative tolerance in comparison between ONNX and PyTorch outputs.
        atol: absolute tolerance in comparison between ONNX and PyTorch outputs.
        remained_onnx_input_idx: If provided, only the specified inputs will be passed
            to the ONNX model. Supply a list when there are unused inputs in the model.
            Since unused inputs will be removed in the exported ONNX model, supplying
            all inputs will cause an error on unexpected inputs. This parameter tells
            the verifier which inputs to pass into the ONNX model.
        acceptable_error_percentage: acceptable percentage of element mismatches in comparison.
            It should be a float of value between 0.0 and 1.0.
    """

    flatten: bool = True
    ignore_none: bool = True
    check_shape: bool = True
    check_dtype: bool = True
    backend: OnnxBackend = OnnxBackend.ONNX_RUNTIME_CPU
    rtol: float = 1e-3
    atol: float = 1e-7
    remained_onnx_input_idx: Optional[Sequence[int]] = None
    acceptable_error_percentage: Optional[float] = None

class OnnxTestCaseRepro:
    def __init__(self, repro_dir):
        self.repro_dir = repro_dir
        self.proto, self.inputs, self.outputs = onnx_proto_utils.load_test_case(
            repro_dir
        )

    @classmethod
    @_beartype.beartype
    def create_test_case_repro(
        cls, proto: bytes, inputs, outputs, dir: str, name: Optional[str] = None
    ):
        """Create a repro under "{dir}/test_{name}" for an ONNX test case.

        The test case contains the model and the inputs/outputs data. The directory
        structure is as follows:

        dir
        \u251c\u2500\u2500 test_<name>
        \u2502   \u251c\u2500\u2500 model.onnx
        \u2502   \u2514\u2500\u2500 test_data_set_0
        \u2502       \u251c\u2500\u2500 input_0.pb
        \u2502       \u251c\u2500\u2500 input_1.pb
        \u2502       \u251c\u2500\u2500 output_0.pb
        \u2502       \u2514\u2500\u2500 output_1.pb

        Args:
            proto: ONNX model proto.
            inputs: Inputs to the model.
            outputs: Outputs of the model.
            dir: Directory to save the repro.
            name: Name of the test case. If not specified, a name based on current time
                will be generated.
        Returns:
            Path to the repro.
        """
        if name is None:
            name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        return onnx_proto_utils.export_as_test_case(
            proto,
            _to_numpy(inputs),
            _to_numpy(outputs),
            name,
            dir,
        )

    @_beartype.beartype
    def validate(self, options: VerificationOptions):
        """Run the ONNX test case with options.backend, and compare with the expected outputs.

        Args:
            options: Options for validation.

        Raise:
            AssertionError: if outputs from options.backend and expected outputs are not
                equal up to specified precision.
        """
        onnx_session = _onnx_backend_session(io.BytesIO(self.proto), options.backend)
        run_outputs = onnx_session.run(None, self.inputs)
        if hasattr(onnx_session, "get_outputs"):
            output_names = [o.name for o in onnx_session.get_outputs()]
        elif hasattr(onnx_session, "output_names"):
            output_names = onnx_session.output_names
        else:
            raise ValueError(f"Unknown onnx session type: {type(onnx_session)}")
        expected_outs = [self.outputs[name] for name in output_names]
        _compare_onnx_pytorch_outputs_in_np(run_outputs, expected_outs, options)

