def exp(op, coeff=1, num_steps=None, id=None):
    """Take the exponential of an Operator times a coefficient.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff (float): a scalar coefficient of the operator
        num_steps (int): The number of steps used in the decomposition of the exponential operator,
            also known as the Trotter number. If this value is `None` and the Suzuki-Trotter
            decomposition is needed, an error will be raised.
        id (str): id for the Exp operator. Default is None.

    Returns:
       :class:`Exp`: An :class:`~.operation.Operator` representing an operator exponential.

    .. note::

        This operator supports a batched base, a batched coefficient and a combination of both:

        >>> op = qml.exp(qml.RX([1, 2, 3], wires=0), coeff=4)
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.exp(qml.RX(1, wires=0), coeff=[1, 2, 3])
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.exp(qml.RX([1, 2, 3], wires=0), coeff=[4, 5, 6])
        >>> qml.matrix(op).shape
        (3, 2, 2)

        But it doesn't support batching of operators:

        >>> op = qml.exp([qml.RX(1, wires=0), qml.RX(2, wires=0)], coeff=4)
        AttributeError: 'list' object has no attribute 'batch_size'

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> x = np.array(1.23)
    >>> op = qml.exp(qml.X(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.X(0) @ qml.X(1) + qml.Y(0) @ qml.Y(1)
    >>> isingxy = qml.exp(t, 0.25j * x)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
    True

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit, though it may not be supported by the device and
    may not be differentiable.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     qml.exp(qml.X(0), -0.5j * x)
    ...     return qml.expval(qml.Z(0))
    >>> print(qml.draw(circuit)(1.23))
    0: ──Exp─┤  <Z>

    If the base operator is Hermitian and the coefficient is real, then the ``Exp`` operator
    can be measured as an observable:

    >>> obs = qml.exp(qml.Z(0), 3)
    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit():
    ...     return qml.expval(obs)
    >>> circuit()
    tensor(20.08553692, requires_grad=True)

    """
    return Exp(op, coeff, num_steps=num_steps, id=id)

def exp(op, coeff=1, num_steps=None, id=None):
    """Take the exponential of an Operator times a coefficient.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff (float): a scalar coefficient of the operator
        num_steps (int): The number of steps used in the decomposition of the exponential operator,
            also known as the Trotter number. If this value is `None` and the Suzuki-Trotter
            decomposition is needed, an error will be raised.
        id (str): id for the Exp operator. Default is None.

    Returns:
       :class:`Exp`: An :class:`~.operation.Operator` representing an operator exponential.

    .. note::

        This operator supports a batched base, a batched coefficient and a combination of both:

        >>> op = qml.exp(qml.RX([1, 2, 3], wires=0), coeff=4)
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.exp(qml.RX(1, wires=0), coeff=[1, 2, 3])
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.exp(qml.RX([1, 2, 3], wires=0), coeff=[4, 5, 6])
        >>> qml.matrix(op).shape
        (3, 2, 2)

        But it doesn't support batching of operators:

        >>> op = qml.exp([qml.RX(1, wires=0), qml.RX(2, wires=0)], coeff=4)
        AttributeError: 'list' object has no attribute 'batch_size'

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> x = np.array(1.23)
    >>> op = qml.exp(qml.X(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.X(0) @ qml.X(1) + qml.Y(0) @ qml.Y(1)
    >>> isingxy = qml.exp(t, 0.25j * x)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
    True

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit, though it may not be supported by the device and
    may not be differentiable.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     qml.exp(qml.X(0), -0.5j * x)
    ...     return qml.expval(qml.Z(0))
    >>> print(qml.draw(circuit)(1.23))
    0: ──Exp─┤  <Z>

    If the base operator is Hermitian and the coefficient is real, then the ``Exp`` operator
    can be measured as an observable:

    >>> obs = qml.exp(qml.Z(0), 3)
    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit():
    ...     return qml.expval(obs)
    >>> circuit()
    tensor(20.08553692, requires_grad=True)

    """
    return Exp(op, coeff, num_steps=num_steps, id=id)

