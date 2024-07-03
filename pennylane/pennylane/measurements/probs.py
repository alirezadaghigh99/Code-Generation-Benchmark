def probs(wires=None, op=None) -> "ProbabilityMP":
    r"""Probability of each computational basis state.

    This measurement function accepts either a wire specification or
    an observable. Passing wires to the function
    instructs the QNode to return a flat array containing the
    probabilities :math:`|\langle i | \psi \rangle |^2` of measuring
    the computational basis state :math:`| i \rangle` given the current
    state :math:`| \psi \rangle`.

    Marginal probabilities may also be requested by restricting
    the wires to a subset of the full system; the size of the
    returned array will be ``[2**len(wires)]``.

    .. Note::
        If no wires or observable are given, the probability of all wires is returned.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
        op (Observable or MeasurementValue or Sequence[MeasurementValue]): Observable (with a ``diagonalizing_gates``
            attribute) that rotates the computational basis, or a  ``MeasurementValue``
            corresponding to mid-circuit measurements.

    Returns:
        ProbabilityMP: Measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.probs(wires=[0, 1])

    Executing this QNode:

    >>> circuit()
    array([0.5, 0.5, 0. , 0. ])

    The returned array is in lexicographic order, so corresponds
    to a :math:`50\%` chance of measuring either :math:`|00\rangle`
    or :math:`|01\rangle`.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

        @qml.qnode(dev)
        def circuit():
            qml.Z(0)
            qml.X(1)
            return qml.probs(op=qml.Hermitian(H, wires=0))

    >>> circuit()
    array([0.14644661 0.85355339])

    The returned array is in lexicographic order, so corresponds
    to a :math:`14.6\%` chance of measuring the rotated :math:`|0\rangle` state
    and :math:`85.4\%` of measuring the rotated :math:`|1\rangle` state.

    Note that the output shape of this measurement process depends on whether
    the device simulates qubit or continuous variable quantum systems.
    """
    if isinstance(op, MeasurementValue):
        if len(op.measurements) > 1:
            raise ValueError(
                "Cannot use qml.probs() when measuring multiple mid-circuit measurements collected "
                "using arithmetic operators. To collect probabilities for multiple mid-circuit "
                "measurements, use a list of mid-circuit measurements with qml.probs()."
            )
        return ProbabilityMP(obs=op)

    if isinstance(op, Sequence):
        if not qml.math.is_abstract(op[0]) and not all(
            isinstance(o, MeasurementValue) and len(o.measurements) == 1 for o in op
        ):
            raise qml.QuantumFunctionError(
                "Only sequences of single MeasurementValues can be passed with the op argument. "
                "MeasurementValues manipulated using arithmetic operators cannot be used when "
                "collecting statistics for a sequence of mid-circuit measurements."
            )

        return ProbabilityMP(obs=op)

    if isinstance(op, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
        raise qml.QuantumFunctionError("Hamiltonians are not supported for rotating probabilities.")

    if op is not None and not qml.math.is_abstract(op) and not op.has_diagonalizing_gates:
        raise qml.QuantumFunctionError(
            f"{op} does not define diagonalizing gates : cannot be used to rotate the probability"
        )

    if wires is not None:
        if op is not None:
            raise qml.QuantumFunctionError(
                "Cannot specify the wires to probs if an observable is "
                "provided. The wires for probs will be determined directly from the observable."
            )
        wires = Wires(wires)
    return ProbabilityMP(obs=op, wires=wires)