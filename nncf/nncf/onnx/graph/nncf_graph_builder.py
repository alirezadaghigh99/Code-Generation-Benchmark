class ONNXLayerAttributes(BaseLayerAttributes):
    """
    Every NNCFNode for ONNX backend has a ONNXLayerAttributes.
    If node has weight tensor(-s), information for algorithms about weight is stored in weight_attrs.
    If node has bias tensor, information for algorithms about bias is stored in bias_attrs.
    If node has attributes needed for algorithms, they are stored in node_attrs.
    E.g. 'transA' attribute of Gemm node for Quantization.
    """

    def __init__(
        self,
        weight_attrs: Optional[Dict[int, Dict]] = None,
        bias_attrs: Optional[Dict[str, Any]] = None,
        node_attrs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param weight_attrs: Maps input port id associated with weight to a weight description.
        :param bias_attrs: Maps bias tensor name associated with weight to a weight description.
        :param node_attrs: Maps attribute name to an attribute value.
        """
        self.weight_attrs = weight_attrs if weight_attrs is not None else {}
        self.bias_attrs = bias_attrs if bias_attrs is not None else {}
        self.node_attrs = node_attrs if node_attrs is not None else {}

    def has_weight(self) -> bool:
        return bool(self.weight_attrs)

    def has_bias(self) -> bool:
        return bool(self.bias_attrs)

    def has_node_attrs(self) -> bool:
        return bool(self.node_attrs)

