def equal(
    op1: Union[Operator, MeasurementProcess, QuantumTape],
    op2: Union[Operator, MeasurementProcess, QuantumTape],
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
) -> bool:
    r"""Function for determining operator, measurement, and tape equality.

    .. Warning::

        The ``qml.equal`` function is based on a comparison of the types and attributes of the
        measurements or operators, not their mathematical representations. While mathematical
        comparisons between some classes, such as ``Tensor`` and ``Hamiltonian``,  are supported,
        mathematically equivalent operators defined via different classes may return False when
        compared via ``qml.equal``. To be more thorough would require the matrix forms to be
        calculated, which may drastically increase runtime.

    .. Warning::

        The interfaces and trainability of data within some observables including ``Tensor``,
        ``Hamiltonian``, ``Prod``, ``Sum`` are sometimes ignored, regardless of what the user
        specifies for ``check_interface`` and ``check_trainability``.

    Args:
        op1 (.Operator or .MeasurementProcess or .QuantumTape): First object to compare
        op2 (.Operator or .MeasurementProcess or .QuantumTape): Second object to compare
        check_interface (bool, optional): Whether to compare interfaces. Default: ``True``.
        check_trainability (bool, optional): Whether to compare trainability status. Default: ``True``.
        rtol (float, optional): Relative tolerance for parameters.
        atol (float, optional): Absolute tolerance for parameters.

    Returns:
        bool: ``True`` if the operators, measurement processes, or tapes are equal, else ``False``

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

        You can use the optional arguments to get more specific results:

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

def assert_equal(
    op1: Union[Operator, MeasurementProcess, QuantumTape],
    op2: Union[Operator, MeasurementProcess, QuantumTape],
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
) -> None:
    """Function to assert that two operators, measurements, or tapes are equal

    Args:
        op1 (.Operator or .MeasurementProcess or .QuantumTape): First object to compare
        op2 (.Operator or .MeasurementProcess or .QuantumTape): Second object to compare
        check_interface (bool, optional): Whether to compare interfaces. Default: ``True``.
        check_trainability (bool, optional): Whether to compare trainability status. Default: ``True``.
        rtol (float, optional): Relative tolerance for parameters.
        atol (float, optional): Absolute tolerance for parameters.

    Returns:
        None

    Raises:
        AssertionError: An ``AssertionError`` is raised if the two operators are not equal.

    .. warning::

        This function is still under developement.

    .. seealso::

        :func:`~.equal`

    **Example**

    >>> op1 = qml.RX(np.array(0.12), wires=0)
    >>> op2 = qml.RX(np.array(1.23), wires=0)
    >>> qml.assert_equal(op1, op1)
    AssertionError: op1 and op2 have different data.
    Got (array(0.12),) and (array(1.23),)

    >>> h1 = qml.Hamiltonian([1, 2], [qml.PauliX(0), qml.PauliY(1)])
    >>> h2 = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliY(1)])
    >>> qml.assert_equal(h1, h2)
    AssertionError: op1 and op2 have different operands because op1 and op2 have different scalars. Got 2 and 1

    """

    dispatch_result = _equal(
        op1,
        op2,
        check_interface=check_interface,
        check_trainability=check_trainability,
        atol=atol,
        rtol=rtol,
    )
    if isinstance(dispatch_result, str):
        raise AssertionError(dispatch_result)
    if not dispatch_result:
        raise AssertionError(f"{op1} and {op2} are not equal for an unspecified reason.")

