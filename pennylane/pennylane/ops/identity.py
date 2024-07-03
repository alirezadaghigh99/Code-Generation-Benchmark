class Identity(CVObservable, Operation):
    r"""
    The Identity operator

    The expectation of this observable

    .. math::
        E[I] = \text{Tr}(I \rho)

    .. seealso:: The equivalent short-form alias :class:`~I`

    Args:
        wires (Iterable[Any] or Any): Wire label(s) that the identity acts on.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    Corresponds to the trace of the quantum state, which in exact
    simulators should always be equal to 1.
    """

    num_params = 0
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    grad_method = None
    """Gradient computation method."""

    _queue_category = "_ops"

    ev_order = 1

    def _flatten(self):
        return tuple(), (self.wires, tuple())

    def __init__(self, wires=None, id=None):
        super().__init__(wires=[] if wires is None else wires, id=id)
        self._hyperparameters = {"n_wires": len(self.wires)}
        self._pauli_rep = qml.pauli.PauliSentence({qml.pauli.PauliWord({}): 1.0})

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "I"

    def __repr__(self):
        """String representation."""
        if len(self.wires) == 0:
            return "I()"
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"I('{wire}')"
        return f"I({wire})"

    @property
    def name(self):
        return "Identity"

    @staticmethod
    def compute_eigvals(n_wires=1):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.I.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.I.compute_eigvals())
        [ 1 1]
        """
        return qml.math.ones(2**n_wires)

    @staticmethod
    @lru_cache()
    def compute_matrix(n_wires=1):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Identity.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Identity.compute_matrix())
        [[1. 0.]
         [0. 1.]]
        """
        return qml.math.eye(int(2**n_wires))

    @staticmethod
    @lru_cache()
    def compute_sparse_matrix(n_wires=1):  # pylint: disable=arguments-differ
        return sparse.eye(int(2**n_wires), format="csr")

    def matrix(self, wire_order=None):
        n_wires = len(wire_order) if wire_order else len(self.wires)
        return self.compute_matrix(n_wires=n_wires)

    @staticmethod
    def _heisenberg_rep(p):
        return qml.math.array([1, 0, 0])

    @staticmethod
    def compute_diagonalizing_gates(
        wires, n_wires=1
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Identity.diagonalizing_gates`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on

        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> qml.Identity.compute_diagonalizing_gates(wires=[0])
        []
        """
        return []

    @staticmethod
    def compute_decomposition(wires, n_wires=1):  # pylint:disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Identity.decomposition`.

        Args:
            wires (Any, Wires): A single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.Identity.compute_decomposition(wires=0)
        []

        """
        return []

    @staticmethod
    def identity_op(*params):
        """Alias for matrix representation of the identity operator."""
        return I.compute_matrix(*params)

    def adjoint(self):
        return I(wires=self.wires)

    # pylint: disable=unused-argument
    def pow(self, z):
        return [I(wires=self.wires)]