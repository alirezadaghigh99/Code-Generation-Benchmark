class ExportOptions:
    """Arguments used by :func:`torch.onnx.export`.

    TODO: Adopt this in `torch.onnx.export` api to replace keyword arguments.
    """

    export_params: bool = True
    verbose: bool = False
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
    input_names: Optional[Sequence[str]] = None
    output_names: Optional[Sequence[str]] = None
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX
    opset_version: Optional[int] = None
    do_constant_folding: bool = True
    dynamic_axes: Optional[Mapping[str, Union[Mapping[int, str], Sequence[int]]]] = None
    keep_initializers_as_inputs: Optional[bool] = None
    custom_opsets: Optional[Mapping[str, int]] = None
    export_modules_as_functions: Union[bool, Set[Type[torch.nn.Module]]] = False

