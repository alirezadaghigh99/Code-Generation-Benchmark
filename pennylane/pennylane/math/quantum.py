def reduce_dm(density_matrix, indices, check_state=False, c_dtype="complex128"):
    """Compute the density matrix from a state represented with a density matrix.

    Args:
        density_matrix (tensor_like): 2D or 3D density matrix tensor. This tensor should be of size ``(2**N, 2**N)`` or
            ``(batch_dim, 2**N, 2**N)``, for some integer number of wires``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))`` or ``(batch_dim, 2**len(indices), 2**len(indices))``

    .. seealso:: :func:`pennylane.math.reduce_statevector`, and :func:`pennylane.density_matrix`

    **Example**

    >>> x = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> reduce_dm(x, indices=[0])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> y = [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
    >>> reduce_dm(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> reduce_dm(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.complex128)
    >>> reduce_dm(z, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    >>> x = np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ...               [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    >>> reduce_dm(x, indices=[1])
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])
    """
    density_matrix = cast(density_matrix, dtype=c_dtype)

    if check_state:
        _check_density_matrix(density_matrix)

    if len(np.shape(density_matrix)) == 2:
        batch_dim, dim = None, density_matrix.shape[0]
    else:
        batch_dim, dim = density_matrix.shape[:2]

    num_indices = int(np.log2(dim))
    consecutive_indices = list(range(num_indices))

    # Return the full density matrix if all the wires are given, potentially permuted
    if len(indices) == num_indices:
        return _permute_dense_matrix(density_matrix, consecutive_indices, indices, batch_dim)

    if batch_dim is None:
        density_matrix = qml.math.stack([density_matrix])

    # Compute the partial trace
    traced_wires = [x for x in consecutive_indices if x not in indices]
    density_matrix = partial_trace(density_matrix, traced_wires, c_dtype=c_dtype)

    if batch_dim is None:
        density_matrix = density_matrix[0]

    # Permute the remaining indices of the density matrix
    return _permute_dense_matrix(density_matrix, sorted(indices), indices, batch_dim)

def expectation_value(
    operator_matrix, state_vector, check_state=False, check_operator=False, c_dtype="complex128"
):
    r"""Compute the expectation value of an operator with respect to a pure state.

    The expectation value is the probabilistic expected result of an experiment.
    Given a pure state, i.e., a state which can be represented as a single
    vector :math:`\ket{\psi}` in the Hilbert space, the expectation value of an
    operator :math:`A` can computed as

    .. math::
        \langle A \rangle_\psi = \bra{\psi} A \ket{\psi}


    Args:
        operator_matrix (tensor_like): operator matrix with shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``.
        state_vector (tensor_like): state vector with shape ``(2**N)`` or ``(batch_dim, 2**N)``.
        check_state (bool): if True, the function will check the validity of the state vector
            via its shape and the norm.
        check_operator (bool): if True, the function will check the validity of the operator
            via its shape and whether it is hermitian.
        c_dtype (str): complex floating point precision type.

    Returns:
        float: Expectation value of the operator for the state vector.

    **Example**

    The expectation value for any operator can obtained by passing their matrix representation as an argument.
    For example, for a 2 qubit state, we can compute the expectation value of the operator :math:`Z \otimes I` as


    >>> state_vector = [1/np.sqrt(2), 0, 1/np.sqrt(2), 0]
    >>> operator_matrix = qml.matrix(qml.PauliZ(0), wire_order=[0,1])
    >>> qml.math.expectation_value(operator_matrix, state_vector)
    tensor(-2.23711432e-17+0.j, requires_grad=True)

    .. seealso:: :func:`pennylane.math.fidelity`

    """
    state_vector = cast(state_vector, dtype=c_dtype)
    operator_matrix = cast(operator_matrix, dtype=c_dtype)

    if check_state:
        _check_state_vector(state_vector)

    if check_operator:
        _check_hermitian_operator(operator_matrix)

    if qml.math.shape(operator_matrix)[-1] != qml.math.shape(state_vector)[-1]:
        raise qml.QuantumFunctionError(
            "The operator and the state vector must have the same number of wires."
        )

    # The overlap <psi|A|psi>
    expval = qml.math.einsum(
        "...i,...i->...",
        qml.math.conj(state_vector),
        qml.math.einsum("...ji,...i->...j", operator_matrix, state_vector, optimize="greedy"),
        optimize="greedy",
    )
    return expval

def reduce_statevector(state, indices, check_state=False, c_dtype="complex128"):
    """Compute the density matrix from a state vector.

    Args:
        state (tensor_like): 1D or 2D tensor state vector. This tensor should of size ``(2**N,)``
            or ``(batch_dim, 2**N)``, for some integer value ``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))`` or ``(batch_dim, 2**len(indices), 2**len(indices))``

    .. seealso:: :func:`pennylane.math.reduce_dm` and :func:`pennylane.density_matrix`

    **Example**

    >>> x = np.array([1, 0, 0, 0])
    >>> reduce_statevector(x, indices=[0])
    [[1.+0.j 0.+0.j]
    [0.+0.j 0.+0.j]]

    >>> y = [1, 0, 1, 0] / np.sqrt(2)
    >>> reduce_statevector(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> reduce_statevector(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([1, 0, 0, 0], dtype=tf.complex128)
    >>> reduce_statevector(z, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    >>> x = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    >>> reduce_statevector(x, indices=[1])
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])
    """
    state = cast(state, dtype=c_dtype)

    # Check the format and norm of the state vector
    if check_state:
        _check_state_vector(state)

    if len(np.shape(state)) == 1:
        batch_dim, dim = None, np.shape(state)[0]
    else:
        batch_dim, dim = np.shape(state)[:2]

        # batch dim exists but is unknown; cast to int so that reshaping works
        if batch_dim is None:
            batch_dim = -1

    # Get dimension of the quantum system and reshape
    num_wires = int(np.log2(dim))
    consecutive_wires = list(range(num_wires))

    if batch_dim is None:
        state = qml.math.stack([state])

    state = np.reshape(state, [batch_dim if batch_dim is not None else 1] + [2] * num_wires)

    # Get the system to be traced
    # traced_system = [x + 1 for x in consecutive_wires if x not in indices]

    # trace out the subsystem
    indices1 = ABC[1 : num_wires + 1]
    indices2 = "".join(
        [ABC[num_wires + i + 1] if i in indices else ABC[i + 1] for i in consecutive_wires]
    )
    target = "".join(
        [ABC[i + 1] for i in sorted(indices)] + [ABC[num_wires + i + 1] for i in sorted(indices)]
    )
    density_matrix = einsum(
        f"a{indices1},a{indices2}->a{target}",
        state,
        np.conj(state),
        optimize="greedy",
    )

    # Return the reduced density matrix by using numpy tensor product
    # density_matrix = np.tensordot(state, np.conj(state), axes=(traced_system, traced_system))

    if batch_dim is None:
        density_matrix = np.reshape(density_matrix, (2 ** len(indices), 2 ** len(indices)))
    else:
        density_matrix = np.reshape(
            density_matrix, (batch_dim, 2 ** len(indices), 2 ** len(indices))
        )

    return _permute_dense_matrix(density_matrix, sorted(indices), indices, batch_dim)

