def equal(
    op1: Union[Operator, MeasurementProcess, QuantumTape],
    op2: Union[Operator, MeasurementProcess, QuantumTape],
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
) -> bool:
    r"""Function for determining operator or measurement equality.

    .. Warning::

        The ``qml.equal`` function is based on a comparison of the type and attributes
        of the measurement or operator, not a mathematical representation. While
        comparisons between some classes, such as ``Tensor`` and ``Hamiltonian``, are
        supported, mathematically equivalent operators defined via different classes
        may return False when compared via ``qml.equal``.

        To be more thorough would require the matrix forms to be calculated, which may
        drastically increase runtime.

    .. Warning::

        The kwargs ``check_interface`` and ``check_trainability`` can only be set when
        comparing ``Operation`` objects. Comparisons of ``MeasurementProcess``
        or ``Observable`` objects will use the default value of ``True`` for both, regardless
        of what the user specifies when calling the function. For subclasses of ``SymbolicOp``
        or ``CompositeOp`` with an ``Operation`` as a base, the kwargs will be applied to the base
        comparison.

    Args:
        op1 (.Operator or .MeasurementProcess or .QuantumTape): First object to compare
        op2 (.Operator or .MeasurementProcess or .QuantumTape): Second object to compare
        check_interface (bool, optional): Whether to compare interfaces. Default: ``True``. Not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor`` objects.
        check_trainability (bool, optional): Whether to compare trainability status. Default: ``True``. Not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor`` objects.
        rtol (float, optional): Relative tolerance for parameters. Not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor`` objects.
        atol (float, optional): Absolute tolerance for parameters. Not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor`` objects.

    Returns:
        bool: ``True`` if the operators or measurement processes are equal, else ``False``

    **Example**

    Given two operators or measurement processes, ``qml.equal`` determines their equality.

    >>> op1 = qml.RX(np.array(.12), wires=0)
    >>> op2 = qml.RY(np.array(1.23), wires=0)
    >>> qml.equal(op1, op1), qml.equal(op1, op2)
    (True, False)

    >>> T1 = qml.X(0) @ qml.Y(1)
    >>> T2 = qml.Y(1) @ qml.X(0)
    >>> T3 = qml.X(1) @ qml.Y(0)
    >>> qml.equal(T1, T2), qml.equal(T1, T3)
    (True, False)

    >>> T = qml.X(0) @ qml.Y(1)
    >>> H = qml.Hamiltonian([1], [qml.X(0) @ qml.Y(1)])
    >>> qml.equal(T, H)
    True

    >>> H1 = qml.Hamiltonian([0.5, 0.5], [qml.Z(0) @ qml.Y(1), qml.Y(1) @ qml.Z(0) @ qml.Identity("a")])
    >>> H2 = qml.Hamiltonian([1], [qml.Z(0) @ qml.Y(1)])
    >>> H3 = qml.Hamiltonian([2], [qml.Z(0) @ qml.Y(1)])
    >>> qml.equal(H1, H2), qml.equal(H1, H3)
    (True, False)

    >>> qml.equal(qml.expval(qml.X(0)), qml.expval(qml.X(0)))
    True
    >>> qml.equal(qml.probs(wires=(0,1)), qml.probs(wires=(1,2)))
    False
    >>> qml.equal(qml.classical_shadow(wires=[0,1]), qml.classical_shadow(wires=[0,1]))
    True
    >>> tape1 = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.Z(0))])
    >>> tape2 = qml.tape.QuantumScript([qml.RX(1.2 + 1e-6, wires=0)], [qml.expval(qml.Z(0))])
    >>> qml.equal(tape1, tape2, tol=0, atol=1e-7)
    False
    >>> qml.equal(tape1, tape2, tol=0, atol=1e-5)
    True

    .. details::
        :title: Usage Details

        You can use the optional arguments to get more specific results. Additionally, they are
        applied when comparing the base of ``SymbolicOp`` and ``CompositeOp`` operators such as
        ``Controlled``, ``Pow``, ``SProd``, ``Prod``, etc., if the base is an ``Operation``. These arguments
        are, however, not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor``
        objects.

        Consider the following comparisons:

        >>> op1 = qml.RX(torch.tensor(1.2), wires=0)
        >>> op2 = qml.RX(jax.numpy.array(1.2), wires=0)
        >>> qml.equal(op1, op2)
        False

        >>> qml.equal(op1, op2, check_interface=False, check_trainability=False)
        True

        >>> op3 = qml.RX(np.array(1.2, requires_grad=True), wires=0)
        >>> op4 = qml.RX(np.array(1.2, requires_grad=False), wires=0)
        >>> qml.equal(op3, op4)
        False

        >>> qml.equal(op3, op4, check_trainability=False)
        True

        >>> qml.equal(Controlled(op3, control_wires=1), Controlled(op4, control_wires=1))
        False

        >>> qml.equal(Controlled(op3, control_wires=1), Controlled(op4, control_wires=1), check_trainability=False)
        True
    """

    if isinstance(op2, (Hamiltonian, Tensor)):
        op1, op2 = op2, op1

    dispatch_result = _equal(
        op1,
        op2,
        check_interface=check_interface,
        check_trainability=check_trainability,
        atol=atol,
        rtol=rtol,
    )
    if isinstance(dispatch_result, str):
        return False
    return dispatch_result