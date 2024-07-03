def expand_vector(vector, original_wires, expanded_wires):
    r"""Expand a vector to more wires.

    Args:
        vector (array): :math:`2^n` vector where n = len(original_wires).
        original_wires (Sequence[int]): original wires of vector
        expanded_wires (Union[Sequence[int], int]): expanded wires of vector, can be shuffled
            If a single int m is given, corresponds to list(range(m))

    Returns:
        array: :math:`2^m` vector where m = len(expanded_wires).
    """
    if len(original_wires) == 0:
        val = qml.math.squeeze(vector)
        return val * qml.math.ones(2 ** len(expanded_wires))
    if isinstance(expanded_wires, numbers.Integral):
        expanded_wires = list(range(expanded_wires))

    N = len(original_wires)
    M = len(expanded_wires)
    D = M - N

    len_vector = qml.math.shape(vector)[0]
    qudit_order = int(2 ** (np.log2(len_vector) / N))

    if not set(expanded_wires).issuperset(original_wires):
        raise ValueError("Invalid target subsystems provided in 'original_wires' argument.")

    if qml.math.shape(vector) != (qudit_order**N,):
        raise ValueError(f"Vector parameter must be of length {qudit_order}**len(original_wires)")

    dims = [qudit_order] * N
    tensor = qml.math.reshape(vector, dims)

    if D > 0:
        extra_dims = [qudit_order] * D
        ones = qml.math.ones(qudit_order**D).reshape(extra_dims)
        expanded_tensor = qml.math.tensordot(tensor, ones, axes=0)
    else:
        expanded_tensor = tensor

    wire_indices = [expanded_wires.index(wire) for wire in original_wires]
    wire_indices = np.array(wire_indices)

    # Order tensor factors according to wires
    original_indices = np.array(range(N))
    expanded_tensor = qml.math.moveaxis(
        expanded_tensor, tuple(original_indices), tuple(wire_indices)
    )

    return qml.math.reshape(expanded_tensor, (qudit_order**M,))def pauli_eigs(n):
    r"""Eigenvalues for :math:`A^{\otimes n}`, where :math:`A` is
    Pauli operator, or shares its eigenvalues.

    As an example if n==2, then the eigenvalues of a tensor product consisting
    of two matrices sharing the eigenvalues with Pauli matrices is returned.

    Args:
        n (int): the number of qubits the matrix acts on
    Returns:
        list: the eigenvalues of the specified observable
    """
    if n == 1:
        return np.array([1.0, -1.0])
    return np.concatenate([pauli_eigs(n - 1), -pauli_eigs(n - 1)])def unflatten(flat, model):
    """Wrapper for :func:`_unflatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Raises:
        ValueError: if ``flat`` has more elements than ``model``
    """
    # pylint:disable=len-as-condition
    res, tail = _unflatten(np.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError("Flattened iterable has more elements than the model.")
    return res