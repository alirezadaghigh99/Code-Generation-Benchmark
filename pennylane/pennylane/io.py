def from_qasm(quantum_circuit: str, measurements=False):
    """Loads quantum circuits from a QASM string using the converter in the
    PennyLane-Qiskit plugin.

    **Example:**

    .. code-block:: python

        >>> hadamard_qasm = 'OPENQASM 2.0;' \\
        ...                 'include "qelib1.inc";' \\
        ...                 'qreg q[1];' \\
        ...                 'h q[0];'
        >>> my_circuit = qml.from_qasm(hadamard_qasm)

    The measurements can also be passed directly to the function when creating the
    quantum function, making it possible to create a PennyLane circuit with
    :class:`qml.QNode <pennylane.QNode>`:

    >>> measurements = [qml.var(qml.Y(0))]
    >>> circuit = qml.QNode(qml.from_qasm(hadamard_qasm, measurements), dev)
    >>> circuit()
    [tensor(1., requires_grad=True)]

    .. note::

        The ``measurements`` keyword allows one to add a list of PennyLane measurements
        that will **override** any terminal measurements present in the QASM code,
        so that they are not performed before the operations specified in ``measurements``.

    By default, ``from_qasm`` will remove any measurements that are present in the QASM code.
    If the QASM code contains measurements, set ``measurements=None`` to keep them in the
    output of ``from_qasm``.

    .. warning::

        The current default behaviour of removing measurements in the QASM code is deprecated
        and will be changed in a future release. Starting in version ``0.38``, ``from_qasm``
        will keep the measurements from the QASM code by default. To remove all measurements,
        set ``measurements=[]`` which overrides the existing measurements with an empty list.

    Mid-circuit measurements inside the QASM code can also be interpreted.

    .. code-block:: python

        hadamard_qasm = 'OPENQASM 2.0;' \\
                         'include "qelib1.inc";' \\
                         'qreg q[2];' \\
                         'creg c[2];' \\
                         'h q[0];' \\
                         'measure q[0] -> c[0];' \\
                         'rz(0.24) q[0];' \\
                         'cx q[0], q[1];' \\
                         'measure q -> c;'

        dev = qml.device("default.qubit")
        loaded_circuit = qml.from_qasm(hadamard_qasm, measurements=None)

        @qml.qnode(dev)
        def circuit():
            mid_measure, m0, m1 = loaded_circuit()
            qml.cond(mid_measure == 0, qml.RX)(np.pi / 2, 0)
            return [qml.expval(qml.measure(0))]

    >>> circuit()
    [tensor(0.75, requires_grad=True)]

    You can also load the contents of a QASM file:

    .. code-block:: python

        >>> with open("hadamard_circuit.qasm", "r") as f:
        ...     my_circuit = qml.from_qasm(f.read())

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.Z(0))

    Args:
        quantum_circuit (str): a QASM string containing a valid quantum circuit
        measurements (None | MeasurementProcess | list[MeasurementProcess]): an optional PennyLane
            measurement or list of PennyLane measurements that overrides any terminal measurements
            that may be present in the input circuit. If set to ``None``, existing measurements
            in the input circuit will be used.

    Returns:
        function: the PennyLane template created based on the QASM string

    """
    try:
        plugin_converter = plugin_converters["qasm"].load()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(  # pragma: no cover
            "Failed to load the qasm plugin. Please ensure that the pennylane-qiskit package is installed."
        ) from e

    if measurements is False:
        measurements = []
        if "measure" in quantum_circuit:
            warnings.warn(
                "The current default behaviour of removing measurements in the QASM code "
                "is deprecated. Set measurements=None to keep existing measurements in the QASM "
                "code or set measurements=[] to remove them from the returned circuit. Starting "
                "in version 0.38, measurements=None will be the new default.",
                qml.PennyLaneDeprecationWarning,
            )
    return plugin_converter(quantum_circuit, measurements=measurements)

