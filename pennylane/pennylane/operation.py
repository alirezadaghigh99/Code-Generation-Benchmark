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
    return __use_new_opmath    def compare(self, other):
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
        )    def simplify(self) -> "Operator":  # pylint: disable=unused-argument
        """Reduce the depth of nested operators to the minimum.

        Returns:
            .Operator: simplified operator
        """
        return self    def map_wires(self, wire_map: dict):
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