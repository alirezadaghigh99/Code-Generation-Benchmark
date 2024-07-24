class WeightCompressionParameters:
    """
    Weight compression parameters determine how and what weight should be compressed.

    :param weight_name: Unique weight name.
    :param node_with_weight: Node with weight in the NNCF graph.
    :param weight_port_id: Number of elements in the weight array.
    :param num_weights: Number of elements in the weight array.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param compression_config: Configuration of weight compression for the weight node.
    """

    weight_name: str
    node_with_weight: NNCFNode
    weight_port_id: int
    num_weights: np.uint64
    reduction_axes: Tuple[int, ...]
    compression_config = WeightCompressionConfig()

    def __post_init__(self):
        # Explicitly cast num_weights to avoid overflow on finding total number of weights.
        # The issue happens on Windows, because np.ndarray.size() returns np.int32 and sum of weights is more than 2^32.
        self.num_weights = np.uint64(self.num_weights)

