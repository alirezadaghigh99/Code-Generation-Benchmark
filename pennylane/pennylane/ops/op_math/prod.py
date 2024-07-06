def _swappable_ops(op1, op2, wire_map: dict = None) -> bool:
    """Boolean expression that indicates if op1 and op2 don't have intersecting wires and if they
    should be swapped when sorting them by wire values.

    Args:
        op1 (.Operator): First operator.
        op2 (.Operator): Second operator.
        wire_map (dict): Dictionary containing the wire values as keys and its indexes as values.
            Defaults to None.

    Returns:
        bool: True if operators should be swapped, False otherwise.
    """
    # one is broadcasted onto all wires.
    if not op1.wires:
        return True
    if not op2.wires:
        return False
    wires1 = op1.wires
    wires2 = op2.wires
    if wire_map is not None:
        wires1 = wires1.map(wire_map)
        wires2 = wires2.map(wire_map)
    wires1 = set(wires1)
    wires2 = set(wires2)
    # compare strings of wire labels so that we can compare arbitrary wire labels like 0 and "a"
    return False if wires1 & wires2 else str(wires1.pop()) > str(wires2.pop())

def prod(*ops, id=None, lazy=True):
    """Construct an operator which represents the generalized product of the
    operators provided.

    The generalized product operation represents both the tensor product as
    well as matrix composition. This can be resolved naturally from the wires
    that the given operators act on.

    Args:
        *ops (Union[tuple[~.operation.Operator], Callable]): The operators we would like to multiply.
            Alternatively, a single qfunc that queues operators can be passed to this function.

    Keyword Args:
        id (str or None): id for the product operator. Default is None.
        lazy=True (bool): If ``lazy=False``, a simplification will be performed such that when any of the operators is already a product operator, its operands will be used instead.

    Returns:
        ~ops.op_math.Prod: the operator representing the product.

    .. note::

        This operator supports batched operands:

        >>> op = qml.prod(qml.RX(np.array([1, 2, 3]), wires=0), qml.X(1))
        >>> op.matrix().shape
        (3, 4, 4)

        But it doesn't support batching of operators:

        >>> op = qml.prod(np.array([qml.RX(0.5, 0), qml.RZ(0.3, 0)]), qml.Z(0))
        AttributeError: 'numpy.ndarray' object has no attribute 'wires'

    .. seealso:: :class:`~.ops.op_math.Prod`

    **Example**

    >>> prod_op = prod(qml.X(0), qml.Z(0))
    >>> prod_op
    X(0) @ Z(0)
    >>> prod_op.matrix()
    array([[ 0, -1],
           [ 1,  0]])
    >>> prod_op.simplify()
    -1j * Y(0)
    >>> prod_op.terms()
    ([-1j], [Y(0)])


    You can also create a prod operator by passing a qfunc to prod, like the following:

    >>> def qfunc(x):
    ...     qml.RX(x, 0)
    ...     qml.CNOT([0, 1])
    >>> prod_op = prod(qfunc)(1.1)
    >>> prod_op
    CNOT(wires=[0, 1]) @ RX(1.1, wires=[0])
    """
    ops = tuple(convert_to_opmath(op) for op in ops)
    if len(ops) == 1:
        if isinstance(ops[0], qml.operation.Operator):
            return ops[0]

        fn = ops[0]

        if not callable(fn):
            raise TypeError(f"Unexpected argument of type {type(fn).__name__} passed to qml.prod")

        @wraps(fn)
        def wrapper(*args, **kwargs):
            qs = qml.tape.make_qscript(fn)(*args, **kwargs)
            if len(qs.operations) == 1:
                if qml.QueuingManager.recording():
                    qml.apply(qs[0])
                return qs[0]
            return prod(*qs.operations[::-1], id=id, lazy=lazy)

        return wrapper

    if lazy:
        return Prod(*ops, id=id)

    ops_simp = Prod(
        *itertools.chain.from_iterable([op if isinstance(op, Prod) else [op] for op in ops]),
        id=id,
    )

    for op in ops:
        QueuingManager.remove(op)

    return ops_simp

def prod(*ops, id=None, lazy=True):
    """Construct an operator which represents the generalized product of the
    operators provided.

    The generalized product operation represents both the tensor product as
    well as matrix composition. This can be resolved naturally from the wires
    that the given operators act on.

    Args:
        *ops (Union[tuple[~.operation.Operator], Callable]): The operators we would like to multiply.
            Alternatively, a single qfunc that queues operators can be passed to this function.

    Keyword Args:
        id (str or None): id for the product operator. Default is None.
        lazy=True (bool): If ``lazy=False``, a simplification will be performed such that when any of the operators is already a product operator, its operands will be used instead.

    Returns:
        ~ops.op_math.Prod: the operator representing the product.

    .. note::

        This operator supports batched operands:

        >>> op = qml.prod(qml.RX(np.array([1, 2, 3]), wires=0), qml.X(1))
        >>> op.matrix().shape
        (3, 4, 4)

        But it doesn't support batching of operators:

        >>> op = qml.prod(np.array([qml.RX(0.5, 0), qml.RZ(0.3, 0)]), qml.Z(0))
        AttributeError: 'numpy.ndarray' object has no attribute 'wires'

    .. seealso:: :class:`~.ops.op_math.Prod`

    **Example**

    >>> prod_op = prod(qml.X(0), qml.Z(0))
    >>> prod_op
    X(0) @ Z(0)
    >>> prod_op.matrix()
    array([[ 0, -1],
           [ 1,  0]])
    >>> prod_op.simplify()
    -1j * Y(0)
    >>> prod_op.terms()
    ([-1j], [Y(0)])


    You can also create a prod operator by passing a qfunc to prod, like the following:

    >>> def qfunc(x):
    ...     qml.RX(x, 0)
    ...     qml.CNOT([0, 1])
    >>> prod_op = prod(qfunc)(1.1)
    >>> prod_op
    CNOT(wires=[0, 1]) @ RX(1.1, wires=[0])
    """
    ops = tuple(convert_to_opmath(op) for op in ops)
    if len(ops) == 1:
        if isinstance(ops[0], qml.operation.Operator):
            return ops[0]

        fn = ops[0]

        if not callable(fn):
            raise TypeError(f"Unexpected argument of type {type(fn).__name__} passed to qml.prod")

        @wraps(fn)
        def wrapper(*args, **kwargs):
            qs = qml.tape.make_qscript(fn)(*args, **kwargs)
            if len(qs.operations) == 1:
                if qml.QueuingManager.recording():
                    qml.apply(qs[0])
                return qs[0]
            return prod(*qs.operations[::-1], id=id, lazy=lazy)

        return wrapper

    if lazy:
        return Prod(*ops, id=id)

    ops_simp = Prod(
        *itertools.chain.from_iterable([op if isinstance(op, Prod) else [op] for op in ops]),
        id=id,
    )

    for op in ops:
        QueuingManager.remove(op)

    return ops_simp

