def counts(
    op=None,
    wires=None,
    all_outcomes=False,
) -> "CountsMP":
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning the number of counts for each sample. If no observable is provided then basis state
    samples are returned directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Observable or MeasurementValue or None): a quantum observable object. To get counts
            for mid-circuit measurements, ``op`` should be a ``MeasurementValue``.
        wires (Sequence[int] or int or None): the wires we wish to sample from, ONLY set wires if
            op is None
        all_outcomes(bool): determines whether the returned dict will contain only the observed
            outcomes (default), or whether it will display all possible outcomes for the system

    Returns:
        CountsMP: Measurement process instance

    Raises:
        ValueError: Cannot set wires if an observable is provided

    The samples are drawn from the eigenvalues :math:`\{\lambda_i\}` of the observable.
    The probability of drawing eigenvalue :math:`\lambda_i` is given by
    :math:`p(\lambda_i) = |\langle \xi_i | \psi \rangle|^2`, where :math:`| \xi_i \rangle`
    is the corresponding basis state from the observable's eigenbasis.

    .. note::

        Differentiation of QNodes that return ``counts`` is currently not supported. Please refer to
        :func:`~.pennylane.sample` if differentiability is required.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.Y(0))

    Executing this QNode:

    >>> circuit(0.5)
    {-1: 2, 1: 2}

    If no observable is provided, then the raw basis state samples obtained
    from device are returned (e.g., for a qubit device, samples from the
    computational device are returned). In this case, ``wires`` can be specified
    so that sample results only include measurement results of the qubits of interest.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

    Executing this QNode:

    >>> circuit(0.5)
    {'00': 3, '01': 1}

    By default, outcomes that were not observed will not be included in the dictionary.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.counts()

    Executing this QNode shows only the observed outcomes:

    >>> circuit()
    {'10': 4}

    Passing all_outcomes=True will create a dictionary that displays all possible outcomes:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.counts(all_outcomes=True)

    Executing this QNode shows counts for all states:

    >>> circuit()
    {'00': 0, '01': 0, '10': 4, '11': 0}

    """
    if isinstance(op, MeasurementValue):
        return CountsMP(obs=op, all_outcomes=all_outcomes)

    if isinstance(op, Sequence):
        if not all(isinstance(o, MeasurementValue) and len(o.measurements) == 1 for o in op):
            raise qml.QuantumFunctionError(
                "Only sequences of single MeasurementValues can be passed with the op argument. "
                "MeasurementValues manipulated using arithmetic operators cannot be used when "
                "collecting statistics for a sequence of mid-circuit measurements."
            )

        return CountsMP(obs=op, all_outcomes=all_outcomes)

    if wires is not None:
        if op is not None:
            raise ValueError(
                "Cannot specify the wires to sample if an observable is provided. The wires "
                "to sample will be determined directly from the observable."
            )
        wires = Wires(wires)

    return CountsMP(obs=op, wires=wires, all_outcomes=all_outcomes)def counts(
    op=None,
    wires=None,
    all_outcomes=False,
) -> "CountsMP":
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning the number of counts for each sample. If no observable is provided then basis state
    samples are returned directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Observable or MeasurementValue or None): a quantum observable object. To get counts
            for mid-circuit measurements, ``op`` should be a ``MeasurementValue``.
        wires (Sequence[int] or int or None): the wires we wish to sample from, ONLY set wires if
            op is None
        all_outcomes(bool): determines whether the returned dict will contain only the observed
            outcomes (default), or whether it will display all possible outcomes for the system

    Returns:
        CountsMP: Measurement process instance

    Raises:
        ValueError: Cannot set wires if an observable is provided

    The samples are drawn from the eigenvalues :math:`\{\lambda_i\}` of the observable.
    The probability of drawing eigenvalue :math:`\lambda_i` is given by
    :math:`p(\lambda_i) = |\langle \xi_i | \psi \rangle|^2`, where :math:`| \xi_i \rangle`
    is the corresponding basis state from the observable's eigenbasis.

    .. note::

        Differentiation of QNodes that return ``counts`` is currently not supported. Please refer to
        :func:`~.pennylane.sample` if differentiability is required.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.Y(0))

    Executing this QNode:

    >>> circuit(0.5)
    {-1: 2, 1: 2}

    If no observable is provided, then the raw basis state samples obtained
    from device are returned (e.g., for a qubit device, samples from the
    computational device are returned). In this case, ``wires`` can be specified
    so that sample results only include measurement results of the qubits of interest.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

    Executing this QNode:

    >>> circuit(0.5)
    {'00': 3, '01': 1}

    By default, outcomes that were not observed will not be included in the dictionary.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.counts()

    Executing this QNode shows only the observed outcomes:

    >>> circuit()
    {'10': 4}

    Passing all_outcomes=True will create a dictionary that displays all possible outcomes:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.counts(all_outcomes=True)

    Executing this QNode shows counts for all states:

    >>> circuit()
    {'00': 0, '01': 0, '10': 4, '11': 0}

    """
    if isinstance(op, MeasurementValue):
        return CountsMP(obs=op, all_outcomes=all_outcomes)

    if isinstance(op, Sequence):
        if not all(isinstance(o, MeasurementValue) and len(o.measurements) == 1 for o in op):
            raise qml.QuantumFunctionError(
                "Only sequences of single MeasurementValues can be passed with the op argument. "
                "MeasurementValues manipulated using arithmetic operators cannot be used when "
                "collecting statistics for a sequence of mid-circuit measurements."
            )

        return CountsMP(obs=op, all_outcomes=all_outcomes)

    if wires is not None:
        if op is not None:
            raise ValueError(
                "Cannot specify the wires to sample if an observable is provided. The wires "
                "to sample will be determined directly from the observable."
            )
        wires = Wires(wires)

    return CountsMP(obs=op, wires=wires, all_outcomes=all_outcomes)