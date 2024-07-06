def op_eq(ops):
    """Builds a conditional as a :class:`~.BooleanFn` for evaluating
    if a given operation is equal to the specified operation.

    Args:
        ops (str, class, Operation): string representation, an instance or class of the operation.

    Returns:
        :class:`OpEq <pennylane.noise.conditionals.OpEq>`: A callable object that accepts
        an :class:`~.Operation` and gives a boolean output. It accepts any input from:
        ``Union(str, class, Operation)`` and evaluates to ``True``, if input operation(s)
        is equal to the set of operation(s) specified by ``ops``, based on a comparison of
        the operation type, irrespective of wires.

    **Example**

    One may use ``op_eq`` with a string representation of the name of the operation:

    >>> cond_func = qml.noise.op_eq("RX")
    >>> cond_func(qml.RX(1.23, wires=[0]))
    True
    >>> cond_func(qml.RZ(1.23, wires=[3]))
    False
    >>> cond_func("CNOT")
    False

    Additionally, an instance of :class:`Operation <pennylane.operation.Operation>`
    can also be provided:

    >>> cond_func = qml.noise.op_eq(qml.RX(1.0, "dino"))
    >>> cond_func(qml.RX(1.23, wires=["eve"]))
    True
    >>> cond_func(qml.RY(1.23, wires=["dino"]))
    False
    """
    return OpEq(ops)

