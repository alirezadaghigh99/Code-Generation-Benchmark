def sum(*summands, grouping_type=None, method="rlf", id=None, lazy=True):
    r"""Construct an operator which is the sum of the given operators.

    Args:
        *summands (tuple[~.operation.Operator]): the operators we want to sum together.

    Keyword Args:
        id (str or None): id for the Sum operator. Default is None.
        lazy=True (bool): If ``lazy=False``, a simplification will be performed such that when any
            of the operators is already a sum operator, its operands (summands) will be used instead.
        grouping_type (str): The type of binary relation between Pauli words used to compute
            the grouping. Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
        method (str): The graph coloring heuristic to use in solving minimum clique cover for
            grouping, which can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest
            First). This keyword argument is ignored if ``grouping_type`` is ``None``.

    Returns:
        ~ops.op_math.Sum: The operator representing the sum of summands.

    .. note::

        This operator supports batched operands:

        >>> op = qml.sum(qml.RX(np.array([1, 2, 3]), wires=0), qml.X(1))
        >>> op.matrix().shape
        (3, 4, 4)

        But it doesn't support batching of operators:

        >>> op = qml.sum(np.array([qml.RX(0.4, 0), qml.RZ(0.3, 0)]), qml.Z(0))
        AttributeError: 'numpy.ndarray' object has no attribute 'wires'

    .. note::

        If grouping is requested, the computed groupings are stored as a list of list of indices
        in ``Sum.grouping_indices``. The indices refer to the operators and coefficients returned
        by ``Sum.terms()``, not ``Sum.operands``, as these are not guaranteed to be equivalent.

    .. seealso:: :class:`~.ops.op_math.Sum`

    **Example**

    >>> summed_op = qml.sum(qml.X(0), qml.Z(0))
    >>> summed_op
    X(0) + Z(0)
    >>> summed_op.matrix()
    array([[ 1,  1],
           [ 1, -1]])

    .. details::
        :title: Grouping

        Grouping information can be collected during construction using the ``grouping_type`` and ``method``
        keyword arguments. For example:

        .. code-block:: python

            import pennylane as qml

            a = qml.s_prod(1.0, qml.X(0))
            b = qml.s_prod(2.0, qml.prod(qml.X(0), qml.X(1)))
            c = qml.s_prod(3.0, qml.Z(0))

            op = qml.sum(a, b, c, grouping_type="qwc")

        >>> op.grouping_indices
        ((2,), (0, 1))

        ``grouping_type`` can be ``"qwc"`` (qubit-wise commuting), ``"commuting"``, or ``"anticommuting"``, and
        ``method`` can be ``"rlf"`` or ``"lf"``. To see more details about how these affect grouping, see
        :ref:`Pauli Graph Colouring<graph_colouring>` and :func:`~pennylane.pauli.group_observables`.
    """
    summands = tuple(convert_to_opmath(op) for op in summands)
    if lazy:
        return Sum(*summands, grouping_type=grouping_type, method=method, id=id)

    summands_simp = Sum(
        *itertools.chain.from_iterable([op if isinstance(op, Sum) else [op] for op in summands]),
        grouping_type=grouping_type,
        method=method,
        id=id,
    )

    for op in summands:
        QueuingManager.remove(op)

    return summands_simp

