def active_new_opmath():
    """
    Function that checks if the new arithmetic operator dunders are active

    Returns:
        bool: Returns ``True`` if the new arithmetic operator dunders are active

    **Example**

    >>> qml.operation.active_new_opmath()
    False
    >>> qml.operation.enable_new_opmath()
    >>> qml.operation.active_new_opmath()
    True
    """
    return __use_new_opmath

def compare(self, other):
        r"""Compares with another :class:`~.Hamiltonian`, :class:`~Tensor`, or :class:`~Observable`,
        to determine if they are equivalent.

        Observables/Hamiltonians are equivalent if they represent the same operator
        (their matrix representations are equal), and they are defined on the same wires.

        .. Warning::

            The compare method does **not** check if the matrix representation
            of a :class:`~.Hermitian` observable is equal to an equivalent
            observable expressed in terms of Pauli matrices.
            To do so would require the matrix form of Hamiltonians and Tensors
            be calculated, which would drastically increase runtime.

        Returns:
            (bool): True if equivalent.

        **Examples**

        >>> ob1 = qml.X(0) @ qml.Identity(1)
        >>> ob2 = qml.Hamiltonian([1], [qml.X(0)])
        >>> ob1.compare(ob2)
        True
        >>> ob1 = qml.X(0)
        >>> ob2 = qml.Hermitian(np.array([[0, 1], [1, 0]]), 0)
        >>> ob1.compare(ob2)
        False
        """
        if isinstance(other, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
            return other.compare(self)
        if isinstance(other, (Tensor, Observable)):
            return other._obs_data() == self._obs_data()

        raise ValueError(
            "Can only compare an Observable/Tensor, and a Hamiltonian/Observable/Tensor."
        )

def simplify(self) -> "Operator":  # pylint: disable=unused-argument
        """Reduce the depth of nested operators to the minimum.

        Returns:
            .Operator: simplified operator
        """
        return self

def map_wires(self, wire_map: dict):
        """Returns a copy of the current operator with its wires changed according to the given
        wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .Operator: new operator
        """
        new_op = copy.copy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        if (p_rep := new_op.pauli_rep) is not None:
            new_op._pauli_rep = p_rep.map_wires(wire_map)
        return new_op

def operation_derivative(operation) -> np.ndarray:
    r"""Calculate the derivative of an operation.

    For an operation :math:`e^{i \hat{H} \phi t}`, this function returns the matrix representation
    in the standard basis of its derivative with respect to :math:`t`, i.e.,

    .. math:: \frac{d \, e^{i \hat{H} \phi t}}{dt} = i \phi \hat{H} e^{i \hat{H} \phi t},

    where :math:`\phi` is a real constant.

    Args:
        operation (.Operation): The operation to be differentiated.

    Returns:
        array: the derivative of the operation as a matrix in the standard basis

    Raises:
        ValueError: if the operation does not have a generator or is not composed of a single
            trainable parameter
    """
    generator = qml.matrix(
        qml.generator(operation, format="observable"), wire_order=operation.wires
    )
    return 1j * generator @ operation.matrix()

def heisenberg_expand(self, U, wire_order):
        """Expand the given local Heisenberg-picture array into a full-system one.

        Args:
            U (array[float]): array to expand (expected to be of the dimension ``1+2*self.num_wires``)
            wire_order (Wires): global wire order defining which subspace the operator acts on

        Raises:
            ValueError: if the size of the input matrix is invalid or `num_wires` is incorrect

        Returns:
            array[float]: expanded array, dimension ``1+2*num_wires``
        """

        U_dim = len(U)
        nw = len(self.wires)

        if U.ndim > 2:
            raise ValueError("Only order-1 and order-2 arrays supported.")

        if U_dim != 1 + 2 * nw:
            raise ValueError(f"{self.name}: Heisenberg matrix is the wrong size {U_dim}.")

        if len(wire_order) == 0 or len(self.wires) == len(wire_order):
            # no expansion necessary (U is a full-system matrix in the correct order)
            return U

        if not wire_order.contains_wires(self.wires):
            raise ValueError(
                f"{self.name}: Some observable wires {self.wires} do not exist on this device with wires {wire_order}"
            )

        # get the indices that the operation's wires have on the device
        wire_indices = wire_order.indices(self.wires)

        # expand U into the I, x_0, p_0, x_1, p_1, ... basis
        dim = 1 + len(wire_order) * 2

        def loc(w):
            "Returns the slice denoting the location of (x_w, p_w) in the basis."
            ind = 2 * w + 1
            return slice(ind, ind + 2)

        if U.ndim == 1:
            W = np.zeros(dim)
            W[0] = U[0]
            for k, w in enumerate(wire_indices):
                W[loc(w)] = U[loc(k)]
        elif U.ndim == 2:
            W = np.zeros((dim, dim)) if isinstance(self, Observable) else np.eye(dim)
            W[0, 0] = U[0, 0]

            for k1, w1 in enumerate(wire_indices):
                s1 = loc(k1)
                d1 = loc(w1)

                # first column
                W[d1, 0] = U[s1, 0]
                # first row (for gates, the first row is always (1, 0, 0, ...), but not for observables!)
                W[0, d1] = U[0, s1]

                for k2, w2 in enumerate(wire_indices):
                    W[d1, loc(w2)] = U[s1, loc(k2)]  # block k1, k2 in U goes to w1, w2 in W.
        return W

def disable_new_opmath_cm():
    r"""Allows to use the old operator arithmetic within a
    temporary context using the `with` statement."""

    was_active = qml.operation.active_new_opmath()
    try:
        if was_active:
            disable_new_opmath(warn=False)
        yield
    except Exception as e:
        raise e
    finally:
        if was_active:
            enable_new_opmath(warn=False)
        else:
            disable_new_opmath(warn=False)

def has_gen(obj):
    """Returns ``True`` if an operator has a generator defined."""
    if isinstance(obj, Operator):
        return obj.has_generator
    try:
        obj.generator()
    except (AttributeError, OperatorPropertyUndefined, GeneratorUndefinedError):
        return False
    return True

def enable_new_opmath_cm():
    r"""Allows to use the new operator arithmetic within a
    temporary context using the `with` statement."""

    was_active = qml.operation.active_new_opmath()
    if not was_active:
        enable_new_opmath(warn=False)
    yield
    if was_active:
        enable_new_opmath(warn=False)
    else:
        disable_new_opmath(warn=False)

