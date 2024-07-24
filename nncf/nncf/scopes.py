class IgnoredScope:
    """
    Provides an option to specify portions of model to be excluded from compression.

    The ignored scope defines model sub-graphs that should be excluded from the compression process such as
    quantization, pruning and etc.

    Example:

    ..  code-block:: python

            import nncf

            # Exclude by node name:
            node_names = ['node_1', 'node_2', 'node_3']
            ignored_scope = nncf.IgnoredScope(names=node_names)

            # Exclude using regular expressions:
            patterns = ['node_\\d']
            ignored_scope = nncf.IgnoredScope(patterns=patterns)

            # Exclude by operation type:

            # OpenVINO opset https://docs.openvino.ai/latest/openvino_docs_ops_opset.html
            operation_types = ['Multiply', 'GroupConvolution', 'Interpolate']
            ignored_scope = nncf.IgnoredScope(types=operation_types)

            # ONNX opset https://github.com/onnx/onnx/blob/main/docs/Operators.md
            operation_types = ['Mul', 'Conv', 'Resize']
            ignored_scope = nncf.IgnoredScope(types=operation_types)

    **Note:** Operation types must be specified according to the model framework.

    :param names: List of ignored node names.
    :type names: List[str]
    :param patterns: List of regular expressions that define patterns for names of ignored nodes.
    :type patterns: List[str]
    :param types: List of ignored operation types.
    :type types: List[str]
    :param subgraphs: List of ignored subgraphs.
    :type subgraphs: List[Subgraph]
    :param validate: If set to True, then a RuntimeError will be raised if any ignored scope does not match
      in the model graph.
    :type types: bool
    """

    names: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    subgraphs: List[Subgraph] = field(default_factory=list)
    validate: bool = True

