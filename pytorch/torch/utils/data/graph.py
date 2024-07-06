def traverse_dps(datapipe: DataPipe) -> DataPipeGraph:
    r"""
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    This only looks into the attribute from each DataPipe that is either a
    DataPipe and a Python collection object such as ``list``, ``tuple``,
    ``set`` and ``dict``.

    Args:
        datapipe: the end DataPipe of the graph
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
    cache: Set[int] = set()
    return _traverse_helper(datapipe, only_datapipe=True, cache=cache)

