def _detach_hidden(
    hidden: Union[torch.Tensor, Sequence, Mapping, str, bytes]
) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Cut backpropagation graph.

    Auxillary function to cut the backpropagation graph by detaching the hidden
    vector.
    """
    return apply_to_tensor(hidden, torch.Tensor.detach)

