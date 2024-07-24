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

class Prod(CompositeOp):
    r"""Symbolic operator representing the product of operators.

    Args:
        *factors (tuple[~.operation.Operator]): a tuple of operators which will be multiplied
            together.

    Keyword Args:
        id (str or None): id for the product operator. Default is None.

    .. seealso:: :func:`~.ops.op_math.prod`

    **Example**

    >>> prod_op = Prod(qml.X(0), qml.PauliZ(1))
    >>> prod_op
    X(0) @ Z(1)
    >>> qml.matrix(prod_op, wire_order=prod_op.wires)
    array([[ 0,  0,  1,  0],
           [ 0,  0,  0, -1],
           [ 1,  0,  0,  0],
           [ 0, -1,  0,  0]])
    >>> prod_op.terms()
    ([1.0], [Z(1) @ X(0)])

    .. note::
        When a Prod operator is applied in a circuit, its factors are applied in the reverse order.
        (i.e ``Prod(op1, op2)`` corresponds to :math:`\hat{op}_{1}\cdot\hat{op}_{2}` which indicates
        first applying :math:`\hat{op}_{2}` then :math:`\hat{op}_{1}` in the circuit). We can see this
        in the decomposition of the operator.

    >>> op = Prod(qml.X(0), qml.Z(1))
    >>> op.decomposition()
    [Z(1), X(0)]

    .. details::
        :title: Usage Details

        The Prod operator represents both matrix composition and tensor products
        between operators.

        >>> prod_op = Prod(qml.RZ(1.23, wires=0), qml.X(0), qml.Z(1))
        >>> prod_op.matrix()
        array([[ 0.        +0.j        ,  0.        +0.j        ,
                 0.81677345-0.57695852j,  0.        +0.j        ],
               [ 0.        +0.j        ,  0.        +0.j        ,
                 0.        +0.j        , -0.81677345+0.57695852j],
               [ 0.81677345+0.57695852j,  0.        +0.j        ,
                 0.        +0.j        ,  0.        +0.j        ],
               [ 0.        +0.j        , -0.81677345-0.57695852j,
                 0.        +0.j        ,  0.        +0.j        ]])

        The Prod operation can be used inside a `qnode` as an operation which,
        if parameterized, can be differentiated.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=3)

            @qml.qnode(dev)
            def circuit(theta):
                qml.prod(qml.Z(0), qml.RX(theta, 1))
                return qml.expval(qml.Z(1))

        >>> par = np.array(1.23, requires_grad=True)
        >>> circuit(par)
        tensor(0.33423773, requires_grad=True)
        >>> qml.grad(circuit)(par)
        tensor(-0.9424888, requires_grad=True)

        The Prod operation can also be measured as an observable.
        If the circuit is parameterized, then we can also differentiate through the
        product observable.

        .. code-block:: python

            prod_op = Prod(qml.Z(0), qml.Hadamard(wires=1))
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit(weights):
                qml.RX(weights[0], wires=0)
                return qml.expval(prod_op)

        >>> weights = np.array([0.1], requires_grad=True)
        >>> qml.grad(circuit)(weights)
        array([-0.07059289])

        Note that the :meth:`~Prod.terms` method always simplifies and flattens the operands.

        >>> op = qml.ops.Prod(qml.X(0), qml.sum(qml.Y(0), qml.Z(1)))
        >>> op.terms()
        ([1j, 1.0], [Z(0), Z(1) @ X(0)])

    """

    _op_symbol = "@"
    _math_op = math.prod

    @property
    def is_hermitian(self):
        """Check if the product operator is hermitian.

        Note, this check is not exhaustive. There can be hermitian operators for which this check
        yields false, which ARE hermitian. So a false result only implies a more explicit check
        must be performed.
        """
        for o1, o2 in combinations(self.operands, r=2):
            if qml.wires.Wires.shared_wires([o1.wires, o2.wires]):
                return False
        return all(op.is_hermitian for op in self)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_decomposition(self):
        return True

    @property
    def obs(self):
        r"""Access the operands of a ``Prod`` instance"""
        # This is temporary property to smoothen the transition to the new operator arithmetic system.
        # In particular, the __matmul__ (@ python operator) method between operators now generates Prod instead of Tensor instances.
        warnings.warn(
            "Accessing the terms of a tensor product operator via op.obs is deprecated, please use op.operands instead.",
            qml.PennyLaneDeprecationWarning,
        )
        return self.operands

    def decomposition(self):
        r"""Decomposition of the product operator is given by each factor applied in succession.

        Note that the decomposition is the list of factors returned in reversed order. This is
        to support the intuition that when we write :math:`\hat{O} = \hat{A} \cdot \hat{B}` it is implied
        that :math:`\hat{B}` is applied to the state before :math:`\hat{A}` in the quantum circuit.
        """
        if qml.queuing.QueuingManager.recording():
            return [qml.apply(op) for op in self[::-1]]
        return list(self[::-1])

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        if self.pauli_rep:
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires)

        mats: List[TensorLike] = []
        batched: List[bool] = []  # batched[i] tells if mats[i] is batched or not
        for ops in self.overlapping_ops:
            gen = (
                (
                    (qml.matrix(op) if isinstance(op, qml.ops.Hamiltonian) else op.matrix()),
                    op.wires,
                )
                for op in ops
            )

            reduced_mat, _ = math.reduce_matrices(gen, reduce_func=math.matmul)

            if self.batch_size is not None:
                batched.append(any(op.batch_size is not None for op in ops))
            else:
                batched.append(False)

            mats.append(reduced_mat)

        if self.batch_size is None:
            full_mat = reduce(math.kron, mats)
        else:
            full_mat = qml.math.stack(
                [
                    reduce(math.kron, [m[i] if b else m for m, b in zip(mats, batched)])
                    for i in range(self.batch_size)
                ]
            )
        return math.expand_matrix(full_mat, self.wires, wire_order=wire_order)

    def sparse_matrix(self, wire_order=None):
        if self.pauli_rep:  # Get the sparse matrix from the PauliSentence representation
            return self.pauli_rep.to_mat(wire_order=wire_order or self.wires, format="csr")

        if self.has_overlapping_wires or self.num_wires > MAX_NUM_WIRES_KRON_PRODUCT:
            gen = ((op.sparse_matrix(), op.wires) for op in self)

            reduced_mat, prod_wires = math.reduce_matrices(gen, reduce_func=math.dot)

            wire_order = wire_order or self.wires

            return math.expand_matrix(reduced_mat, prod_wires, wire_order=wire_order)
        mats = (op.sparse_matrix() for op in self)
        full_mat = reduce(sparse_kron, mats)
        return math.expand_matrix(full_mat, self.wires, wire_order=wire_order)

    # pylint: disable=protected-access
    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
        * `"_ops"`
        * `"_measurements"`
        * `None`

        Returns (str or None): "_ops" if the _queue_catagory of all factors is "_ops", else None.
        """
        return "_ops" if all(op._queue_category == "_ops" for op in self) else None

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_adjoint(self):
        return True

    def adjoint(self):
        return Prod(*(qml.adjoint(factor) for factor in self[::-1]))

    @property
    def arithmetic_depth(self) -> int:
        return 1 + max(factor.arithmetic_depth for factor in self)

    def _build_pauli_rep(self):
        """PauliSentence representation of the Product of operations."""
        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps)
        return None

    def _simplify_factors(self, factors: Tuple[Operator]) -> Tuple[complex, Operator]:
        """Reduces the depth of nested factors and groups identical factors.

        Returns:
            Tuple[complex, List[~.operation.Operator]: tuple containing the global phase and a list
            of the simplified factors
        """
        new_factors = _ProductFactorsGrouping()

        for factor in factors:
            simplified_factor = factor.simplify()
            new_factors.add(factor=simplified_factor)
        new_factors.remove_factors(wires=self.wires)
        return new_factors.global_phase, new_factors.factors

    def simplify(self) -> Union["Prod", Sum]:
        r"""
        Transforms any nested Prod instance into the form :math:`\sum c_i O_i` where
        :math:`c_i` is a scalar coefficient and :math:`O_i` is a single PL operator
        or pure product of single PL operators.
        """
        # try using pauli_rep:
        if pr := self.pauli_rep:
            pr.simplify()
            return pr.operation(wire_order=self.wires)

        global_phase, factors = self._simplify_factors(factors=self.operands)

        factors = list(itertools.product(*factors))
        if len(factors) == 1:
            factor = factors[0]
            if len(factor) == 0:
                op = qml.Identity(self.wires)
            else:
                op = factor[0] if len(factor) == 1 else Prod(*factor)
            return op if global_phase == 1 else qml.s_prod(global_phase, op)

        factors = [Prod(*factor).simplify() if len(factor) > 1 else factor[0] for factor in factors]
        op = Sum(*factors).simplify()
        return op if global_phase == 1 else qml.s_prod(global_phase, op).simplify()

    @classmethod
    def _sort(cls, op_list, wire_map: dict = None) -> List[Operator]:
        """Insertion sort algorithm that sorts a list of product factors by their wire indices, taking
        into account the operator commutivity.

        Args:
            op_list (List[.Operator]): list of operators to be sorted
            wire_map (dict): Dictionary containing the wire values as keys and its indexes as values.
                Defaults to None.

        Returns:
            List[.Operator]: sorted list of operators
        """

        if isinstance(op_list, tuple):
            op_list = list(op_list)

        for i in range(1, len(op_list)):
            key_op = op_list[i]

            j = i - 1
            while j >= 0 and _swappable_ops(op1=op_list[j], op2=key_op, wire_map=wire_map):
                op_list[j + 1] = op_list[j]
                j -= 1
            op_list[j + 1] = key_op

        return op_list

    def terms(self):
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`

        **Example**

        >>> op = X(0) @ (0.5 * X(1) + X(2))
        >>> op.terms()
        ([0.5, 1.0],
         [X(1) @ X(0),
          X(2) @ X(0)])

        """
        # try using pauli_rep:
        if pr := self.pauli_rep:
            with qml.QueuingManager.stop_recording():
                ops = [pauli.operation() for pauli in pr.keys()]
            return list(pr.values()), ops

        with qml.QueuingManager.stop_recording():
            global_phase, factors = self._simplify_factors(factors=self.operands)
            factors = list(itertools.product(*factors))

            factors = [
                Prod(*factor).simplify() if len(factor) > 1 else factor[0] for factor in factors
            ]

        # harvest coeffs and ops
        coeffs = []
        ops = []
        for factor in factors:
            if isinstance(factor, SProd):
                coeffs.append(global_phase * factor.scalar)
                ops.append(factor.base)
            else:
                coeffs.append(global_phase)
                ops.append(factor)
        return coeffs, ops

    @property
    def coeffs(self):
        r"""
        Scalar coefficients of the operator when flattened out.

        This is a deprecated attribute, please use :meth:`~Prod.terms` instead.

        .. seealso:: :attr:`~Prod.ops`, :class:`~Prod.pauli_rep`"""
        warnings.warn(
            "Prod.coeffs is deprecated and will be removed in future releases. You can access both (coeffs, ops) via op.terms(). Also consider op.operands.",
            qml.PennyLaneDeprecationWarning,
        )
        coeffs, _ = self.terms()
        return coeffs

    @property
    def ops(self):
        r"""
        Operator terms without scalar coefficients of the operator when flattened out.

        This is a deprecated attribute, please use :meth:`~Prod.terms` instead.

        .. seealso:: :attr:`~Prod.coeffs`, :class:`~Prod.pauli_rep`"""
        warnings.warn(
            "Prod.ops is deprecated and will be removed in future releases. You can access both (coeffs, ops) via op.terms() Also consider op.operands.",
            qml.PennyLaneDeprecationWarning,
        )
        _, ops = self.terms()
        return ops

