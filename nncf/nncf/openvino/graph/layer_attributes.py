class OVLayerAttributes(BaseLayerAttributes):
    """
    This class stores additional information about nodes that needs to be processed during compression.
    """

    def __init__(
        self,
        constant_attributes: Dict[int, Any],
        layer_attributes: Optional[BaseLayerAttributes] = None,
        inputs_attributes: Optional[Dict[Any, Any]] = None,
    ):
        """
        :param constant_attributes: Map of weights port ID to corresponding const attributes.
        :param layer_attributes: Map of weights port ID to corresponding common layer attributes.
        :param inputs_attributes: Activation attributes.
        """
        self._constant_attributes = constant_attributes
        self._layer_attributes = layer_attributes
        self._inputs_attributes = inputs_attributes

    @property
    def constant_attributes(self) -> Dict[int, Any]:
        return self._constant_attributes

    @property
    def layer_attributes(self) -> Optional[BaseLayerAttributes]:
        return self._layer_attributes

    @property
    def input_attributes(self) -> Optional[Dict[Any, Any]]:
        return self._inputs_attributes

    def get_const_port_ids(self) -> List[int]:
        """
        Returns indices of input ports corresponding to the constant nodes.

        :returns: List of input port indices with constants.
        """
        if self._constant_attributes is not None:
            return list(self._constant_attributes.keys())
        return []

