def extract_tensors(x):
    """Iterate through an iterable, and extract any PennyLane
    tensors that appear.

    Args:
        x (.tensor or Sequence): an input tensor or sequence

    Yields:
        tensor: the next tensor in the sequence. If the input was a single
        tensor, than the tensor is yielded and the iterator completes.

    **Example**

    >>> from pennylane import numpy as np
    >>> import numpy as onp
    >>> iterator = np.extract_tensors([0.1, np.array(0.1), "string", onp.array(0.5)])
    >>> list(iterator)
    [tensor(0.1, requires_grad=True)]
    """
    if isinstance(x, tensor):
        # If the item is a tensor, return it
        yield x
    elif isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        # If the item is a sequence, recursively look through its
        # elements for tensors.
        # NOTE: we choose to branch on Sequence here and not Iterable,
        # as NumPy arrays are not Sequences.
        for item in x:
            yield from extract_tensors(item)

