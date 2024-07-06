def to_mat(self, wire_order=None, format="dense", coeff=1.0):
        """Returns the matrix representation.

        Keyword Args:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix. It is "dense" by default. Use "csr" for sparse.
            coeff (float): Coefficient multiplying the resulting matrix.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the Pauli word.

        Raises:
            ValueError: Can't get the matrix of an empty PauliWord.
        """
        wire_order = self.wires if wire_order is None else Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise ValueError(
                "Can't get the matrix for the specified wire order because it "
                f"does not contain all the Pauli word's wires {self.wires}"
            )

        if len(self) == 0:
            n = len(wire_order) if wire_order is not None else 0
            return (
                np.diag([coeff] * 2**n)
                if format == "dense"
                else coeff * sparse.eye(2**n, format=format, dtype="complex128")
            )

        if format == "dense":
            return coeff * reduce(math.kron, (mat_map[self[w]] for w in wire_order))

        return self._to_sparse_mat(wire_order, coeff)

def hamiltonian(self, wire_order=None):
        """Return :class:`~pennylane.Hamiltonian` representing the PauliWord."""
        if len(self) == 0:
            if wire_order in (None, [], Wires([])):
                raise ValueError("Can't get the Hamiltonian for an empty PauliWord.")
            return qml.Hamiltonian([1], [Identity(wires=wire_order)])

        obs = [_make_operation(op, wire) for wire, op in self.items()]
        return qml.Hamiltonian([1], [obs[0] if len(obs) == 1 else Tensor(*obs)])

