def adjoint(fn, lazy=True):
    """Create the adjoint of an Operator or a function that applies the adjoint of the provided function.
    :func:`~.qjit` compatible.

    Args:
        fn (function or :class:`~.operation.Operator`): A single operator or a quantum function that
            applies quantum operations.

    Keyword Args:
        lazy=True (bool): If the transform is behaving lazily, all operations are wrapped in a ``Adjoint`` class
            and handled later. If ``lazy=False``, operation-specific adjoint decompositions are first attempted.
            Setting ``lazy=False`` is not supported when used with :func:`~.qjit`.

    Returns:
        (function or :class:`~.operation.Operator`): If an Operator is provided, returns an Operator that is the adjoint.
        If a function is provided, returns a function with the same call signature that returns the Adjoint of the
        provided function.

    .. note::

        The adjoint and inverse are identical for unitary gates, but not in general. For example, quantum channels and
        observables may have different adjoint and inverse operators.

    .. note::

        When used with :func:`~.qjit`, this function only supports the Catalyst compiler.
        See :func:`catalyst.adjoint` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    .. note::

        This function supports a batched operator:

        >>> op = qml.adjoint(qml.RX([1, 2, 3], wires=0))
        >>> qml.matrix(op).shape
        (3, 2, 2)

        But it doesn't support batching of operators:

        >>> op = qml.adjoint([qml.RX(1, wires=0), qml.RX(2, wires=0)])
        ValueError: The object [RX(1, wires=[0]), RX(2, wires=[0])] of type <class 'list'> is not callable.
        This error might occur if you apply adjoint to a list of operations instead of a function or template.

    .. seealso:: :class:`~.ops.op_math.Adjoint` and :meth:`.Operator.adjoint`

    **Example**

    The adjoint transform can accept a single operator.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit2(y):
    ...     qml.adjoint(qml.RY(y, wires=0))
    ...     return qml.expval(qml.Z(0))
    >>> print(qml.draw(circuit2)("y"))
    0: ──RY(y)†─┤  <Z>
    >>> print(qml.draw(circuit2, expansion_strategy="device")(0.1))
    0: ──RY(-0.10)─┤  <Z>

    The adjoint transforms can also be used to apply the adjoint of
    any quantum function.  In this case, ``adjoint`` accepts a single function and returns
    a function with the same call signature.

    We can create a QNode that applies the ``my_ops`` function followed by its adjoint:

    .. code-block:: python3

        def my_ops(a, wire):
            qml.RX(a, wires=wire)
            qml.SX(wire)

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(a):
            my_ops(a, wire=0)
            qml.adjoint(my_ops)(a, wire=0)
            return qml.expval(qml.Z(0))

    Printing this out, we can see that the inverse quantum
    function has indeed been applied:

    >>> print(qml.draw(circuit)(0.2))
    0: ──RX(0.20)──SX──SX†──RX(0.20)†─┤  <Z>

    **Example with compiler**

    The adjoint used in a compilation context can be applied on control flow.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def workflow(theta, n, wires):
            def func():
                @qml.for_loop(0, n, 1)
                def loop_fn(i):
                    qml.RX(theta, wires=wires)

                loop_fn()
            qml.adjoint(func)()
            return qml.probs()

    >>> workflow(jnp.pi/2, 3, 0)
    array([0.5, 0.5])

    .. warning::

        The Catalyst adjoint function does not support performing the adjoint
        of quantum functions that contain mid-circuit measurements.

    .. details::
        :title: Lazy Evaluation

        When ``lazy=False``, the function first attempts operation-specific decomposition of the
        adjoint via the :meth:`.Operator.adjoint` method. Only if an Operator doesn't have
        an :meth:`.Operator.adjoint` method is the object wrapped with the :class:`~.ops.op_math.Adjoint`
        wrapper class.

        >>> qml.adjoint(qml.Z(0), lazy=False)
        Z(0)
        >>> qml.adjoint(qml.RX, lazy=False)(1.0, wires=0)
        RX(-1.0, wires=[0])
        >>> qml.adjoint(qml.S, lazy=False)(0)
        Adjoint(S)(wires=[0])

    """
    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.adjoint(fn, lazy=lazy)
    if qml.math.is_abstract(fn):
        return Adjoint(fn)
    return create_adjoint_op(fn, lazy)

