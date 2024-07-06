def get_const_value(const_node: ov.Node, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Returns the constant tensor for the node.

    :param const_node: OpenVINO node.
    :param dtype: Destination type.
    :return: The constant value.
    """
    if dtype is None:
        return const_node.data
    return const_node.get_data(dtype=dtype)

