class PauliRot(Operation):
    r"""
    Arbitrary Pauli word rotation.

    .. math::

        RP(\theta, P) = \exp\left(-i \frac{\theta}{2} P\right)

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\theta}f(RP(\theta)) = \frac{1}{2}\left[f(RP(\theta +\pi/2)) - f(RP(\theta-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`RP(\theta)`.

    .. note::

        If the ``PauliRot`` gate is not supported on the targeted device, PennyLane
        will decompose the gate using :class:`~.RX`, :class:`~.Hadamard`, :class:`~.RZ`
        and :class:`~.CNOT` gates.

    Args:
        theta (float): rotation angle :math:`\theta`
        pauli_word (string): the Pauli word defining the rotation
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.PauliRot(0.5, 'X',  wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> print(example_circuit())
    0.8775825618903724
    """

    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    do_check_domain = False
    grad_method = "A"
    parameter_frequencies = [(1,)]

    _ALLOWED_CHARACTERS = "IXYZ"

    _PAULI_CONJUGATION_MATRICES = {
        "X": Hadamard.compute_matrix(),
        "Y": RX.compute_matrix(np.pi / 2),
        "Z": np.array([[1, 0], [0, 1]]),
    }

    @classmethod
    def _primitive_bind_call(cls, theta, pauli_word, wires=None, id=None):
        return super()._primitive_bind_call(theta, pauli_word=pauli_word, wires=wires, id=id)

    def __init__(self, theta, pauli_word, wires=None, id=None):
        super().__init__(theta, wires=wires, id=id)
        self.hyperparameters["pauli_word"] = pauli_word

        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed. '
                "Allowed characters are I, X, Y and Z"
            )

        num_wires = 1 if isinstance(wires, int) else len(wires)

        if not len(pauli_word) == num_wires:
            raise ValueError(
                f"The given Pauli word has length {len(pauli_word)}, length "
                f"{num_wires} was expected for wires {wires}"
            )

    def __repr__(self):
        return f"PauliRot({self.data[0]}, {self.hyperparameters['pauli_word']}, wires={self.wires.tolist()})"

    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> op = qml.PauliRot(0.1, "XYY", wires=(0,1,2))
        >>> op.label()
        'RXYY'
        >>> op.label(decimals=2)
        'RXYY\n(0.10)'
        >>> op.label(base_label="PauliRot")
        'PauliRot\n(0.10)'

        """
        pauli_word = self.hyperparameters["pauli_word"]
        op_label = base_label or ("R" + pauli_word)

        # TODO[dwierichs]: Implement a proper label for parameter-broadcasted operators
        if decimals is not None and self.batch_size is None:
            param_string = f"\n({qml.math.asarray(self.parameters[0]):.{decimals}f})"
            op_label += param_string

        return op_label

    @staticmethod
    def _check_pauli_word(pauli_word):
        """Check that the given Pauli word has correct structure.

        Args:
            pauli_word (str): Pauli word to be checked

        Returns:
            bool: Whether the Pauli word has correct structure.
        """
        return all(pauli in PauliRot._ALLOWED_CHARACTERS for pauli in set(pauli_word))

    @staticmethod
    def compute_matrix(theta, pauli_word):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PauliRot.matrix`


        Args:
            theta (tensor_like or float): rotation angle
            pauli_word (str): string representation of Pauli word

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.PauliRot.compute_matrix(0.5, 'X')
        [[9.6891e-01+4.9796e-18j 2.7357e-17-2.4740e-01j]
         [2.7357e-17-2.4740e-01j 9.6891e-01+4.9796e-18j]]
        """
        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed. '
                "Allowed characters are I, X, Y and Z"
            )

        interface = qml.math.get_interface(theta)

        if interface == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        # Simplest case is if the Pauli is the identity matrix
        if set(pauli_word) == {"I"}:
            exp = qml.math.exp(-0.5j * theta)
            iden = qml.math.eye(2 ** len(pauli_word), like=theta)
            if qml.math.get_interface(theta) == "tensorflow":
                iden = qml.math.cast_like(iden, 1j)
            if qml.math.get_interface(theta) == "torch":
                td = exp.device
                iden = iden.to(td)

            if qml.math.ndim(theta) == 0:
                return exp * iden

            return qml.math.stack([e * iden for e in exp])

        # We first generate the matrix excluding the identity parts and expand it afterwards.
        # To this end, we have to store on which wires the non-identity parts act
        non_identity_wires, non_identity_gates = zip(
            *[(wire, gate) for wire, gate in enumerate(pauli_word) if gate != "I"]
        )

        multi_Z_rot_matrix = MultiRZ.compute_matrix(theta, len(non_identity_gates))

        # now we conjugate with Hadamard and RX to create the Pauli string
        conjugation_matrix = functools.reduce(
            qml.math.kron,
            [PauliRot._PAULI_CONJUGATION_MATRICES[gate] for gate in non_identity_gates],
        )
        if interface == "tensorflow":
            conjugation_matrix = qml.math.cast_like(conjugation_matrix, 1j)
        # Note: we use einsum with reverse arguments here because it is not multi-dispatched
        # and the tensordot containing multi_Z_rot_matrix should decide about the interface
        return expand_matrix(
            qml.math.einsum(
                "...jk,ij->...ik",
                qml.math.tensordot(multi_Z_rot_matrix, conjugation_matrix, axes=[[-1], [0]]),
                qml.math.conj(conjugation_matrix),
            ),
            non_identity_wires,
            list(range(len(pauli_word))),
        )

    def generator(self):
        pauli_word = self.hyperparameters["pauli_word"]
        wire_map = {w: i for i, w in enumerate(self.wires)}

        return qml.Hamiltonian(
            [-0.5], [qml.pauli.string_to_pauli_word(pauli_word, wire_map=wire_map)]
        )

    @staticmethod
    def compute_eigvals(theta, pauli_word):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PauliRot.eigvals`


        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.PauliRot.compute_eigvals(torch.tensor(0.5), "X")
        tensor([0.9689-0.2474j, 0.9689+0.2474j])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        # Identity must be treated specially because its eigenvalues are all the same
        if set(pauli_word) == {"I"}:
            exp = qml.math.exp(-0.5j * theta)
            ones = qml.math.ones(2 ** len(pauli_word), like=theta)
            if qml.math.get_interface(theta) == "tensorflow":
                ones = qml.math.cast_like(ones, 1j)

            if qml.math.ndim(theta) == 0:
                return exp * ones

            return qml.math.tensordot(exp, ones, axes=0)

        return MultiRZ.compute_eigvals(theta, len(pauli_word))

    @staticmethod
    def compute_decomposition(theta, wires, pauli_word):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.PauliRot.decomposition`.

        Args:
            theta (float): rotation angle :math:`\theta`
            wires (Iterable, Wires): the wires the operation acts on
            pauli_word (string): the Pauli word defining the rotation

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PauliRot.compute_decomposition(1.2, "XY", wires=(0,1))
        [Hadamard(wires=[0]),
        RX(1.5707963267948966, wires=[1]),
        MultiRZ(1.2, wires=[0, 1]),
        Hadamard(wires=[0]),
        RX(-1.5707963267948966, wires=[1])]

        """
        if isinstance(wires, int):  # Catch cases when the wire is passed as a single int.
            wires = [wires]

        # Check for identity and do nothing
        if set(pauli_word) == {"I"}:
            return [qml.GlobalPhase(phi=theta / 2)]

        active_wires, active_gates = zip(
            *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
        )

        ops = []
        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(Hadamard(wires=[wire]))
            elif gate == "Y":
                ops.append(RX(np.pi / 2, wires=[wire]))

        ops.append(MultiRZ(theta, wires=list(active_wires)))

        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(Hadamard(wires=[wire]))
            elif gate == "Y":
                ops.append(RX(-np.pi / 2, wires=[wire]))
        return ops

    def adjoint(self):
        return PauliRot(-self.parameters[0], self.hyperparameters["pauli_word"], wires=self.wires)

    def pow(self, z):
        return [PauliRot(self.data[0] * z, self.hyperparameters["pauli_word"], wires=self.wires)]

class PCPhase(Operation):
    r"""PCPhase(phi, dim, wires)
    A projector-controlled phase gate.

    This gate applies a complex phase :math:`e^{i\phi}` to the first :math:`dim`
    basis vectors of the input state while applying a complex phase :math:`e^{-i \phi}`
    to the remaining basis vectors. For example, consider the 2-qubit case where :math:`dim = 3`:

    .. math:: \Pi(\phi) = \begin{bmatrix}
                e^{i\phi} & 0 & 0 & 0 \\
                0 & e^{i\phi} & 0 & 0 \\
                0 & 0 & e^{i\phi} & 0 \\
                0 & 0 & 0 & e^{-i\phi}
            \end{bmatrix}.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    Args:
        phi (float): rotation angle :math:`\phi`
        dim (int): the dimension of the subspace
        wires (Iterable[int, str], Wires): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    **Example:**

    We can define a circuit using :class:`~.PCPhase` as follows:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...     qml.PCPhase(0.27, dim = 2, wires=range(2))
    ...     return qml.state()

    The resulting operation applies a complex phase :math:`e^{0.27i}` to the first :math:`dim = 2`
    basis vectors and :math:`e^{-0.27i}` to the remaining basis vectors.

    >>> print(np.round(qml.matrix(example_circuit)(),2))
    [[0.96+0.27j 0.  +0.j   0.  +0.j   0.  +0.j  ]
     [0.  +0.j   0.96+0.27j 0.  +0.j   0.  +0.j  ]
     [0.  +0.j   0.  +0.j   0.96-0.27j 0.  +0.j  ]
     [0.  +0.j   0.  +0.j   0.  +0.j   0.96-0.27j]]

    We can also choose a different :math:`dim` value to apply the phase shift to a different set of
    basis vectors as follows:

    >>> pc_op = qml.PCPhase(1.23, dim=3, wires=[1, 2])
    >>> print(np.round(qml.matrix(pc_op),2))
    [[0.33+0.94j 0.  +0.j   0.  +0.j   0.  +0.j  ]
     [0.  +0.j   0.33+0.94j 0.  +0.j   0.  +0.j  ]
     [0.  +0.j   0.  +0.j   0.33+0.94j 0.  +0.j  ]
     [0.  +0.j   0.  +0.j   0.  +0.j   0.33-0.94j]]
    """

    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""
    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(2,)]

    def generator(self):
        dim, shape = self.hyperparameters["dimension"]
        mat = np.diag([1 if index < dim else -1 for index in range(shape)])
        return qml.Hermitian(mat, wires=self.wires)

    def _flatten(self):
        hyperparameter = (("dim", self.hyperparameters["dimension"][0]),)
        return tuple(self.data), (self.wires, hyperparameter)

    def __init__(self, phi, dim, wires, id=None):
        wires = wires if isinstance(wires, Wires) else Wires(wires)

        if not (isinstance(dim, int) and (dim <= 2 ** len(wires))):
            raise ValueError(
                f"The projected dimension {dim} must be an integer that is less than or equal to "
                f"the max size of the matrix {2 ** len(wires)}. Try adding more wires."
            )

        super().__init__(phi, wires=wires, id=id)
        self.hyperparameters["dimension"] = (dim, 2 ** len(wires))

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        """Get the matrix representation of Pi-controlled phase unitary."""
        phi = params[0]
        d, t = hyperparams["dimension"]

        if qml.math.get_interface(phi) == "tensorflow":
            p = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            minus_p = qml.math.exp(-1j * qml.math.cast_like(phi, 1j))
            zeros = qml.math.zeros_like(p)

            columns = []
            for i in range(t):
                columns.append(
                    [p if j == i else zeros for j in range(t)]
                    if i < d
                    else [minus_p if j == i else zeros for j in range(t)]
                )
            r = qml.math.stack(columns, like="tensorflow", axis=-2)
            return r

        arg = 1j * phi
        prefactors = qml.math.array([1 if index < d else -1 for index in range(t)], like=phi)

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * prefactors))

        diags = qml.math.exp(qml.math.outer(arg, prefactors))
        return qml.math.stack([qml.math.diag(d) for d in diags])

    @staticmethod
    def compute_eigvals(*params, **hyperparams):
        """Get the eigvals for the Pi-controlled phase unitary."""
        phi = params[0]
        d, t = hyperparams["dimension"]

        if qml.math.get_interface(phi) == "tensorflow":
            phase = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            minus_phase = qml.math.exp(-1j * qml.math.cast_like(phi, 1j))
            return stack_last([phase if index < d else minus_phase for index in range(t)])

        arg = 1j * phi
        prefactors = qml.math.array([1 if index < d else -1 for index in range(t)], like=phi)

        if qml.math.ndim(phi) == 0:
            product = arg * prefactors
        else:
            product = qml.math.outer(arg, prefactors)
        return qml.math.exp(product)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparams):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyper-parameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """
        phi = params[0]
        k, n = hyperparams["dimension"]

        def _get_op_from_binary_rep(binary_rep, theta, wires):
            if len(binary_rep) == 1:
                op = (
                    PhaseShift(theta, wires[0])
                    if int(binary_rep)
                    else PauliX(wires[0]) @ PhaseShift(theta, wires[0]) @ PauliX(wires[0])
                )
            else:
                base_op = (
                    PhaseShift(theta, wires[-1])
                    if int(binary_rep[-1])
                    else PauliX(wires[-1]) @ PhaseShift(theta, wires[-1]) @ PauliX(wires[-1])
                )
                op = qml.ctrl(
                    base_op, control=wires[:-1], control_values=[int(i) for i in binary_rep[:-1]]
                )
            return op

        n_log2 = int(np.log2(n))
        positive_binary_reps = [bin(_k)[2:].zfill(n_log2) for _k in range(k)]
        negative_binary_reps = [bin(_k)[2:].zfill(n_log2) for _k in range(k, n)]

        positive_ops = [
            _get_op_from_binary_rep(br, phi, wires=wires) for br in positive_binary_reps
        ]
        negative_ops = [
            _get_op_from_binary_rep(br, -1 * phi, wires=wires) for br in negative_binary_reps
        ]

        return positive_ops + negative_ops

    def adjoint(self):
        """Computes the adjoint of the operator."""
        phi = self.parameters[0]
        dim, _ = self.hyperparameters["dimension"]
        return PCPhase(-1 * phi, dim=dim, wires=self.wires)

    def pow(self, z):
        """Computes the operator raised to z."""
        phi = self.parameters[0]
        dim, _ = self.hyperparameters["dimension"]
        return [PCPhase(phi * z, dim=dim, wires=self.wires)]

    def simplify(self):
        """Simplifies the operator if possible."""
        phi = self.parameters[0] % (2 * np.pi)
        dim, _ = self.hyperparameters["dimension"]

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return PCPhase(phi, dim=dim, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        """The label of the operator when displayed in a circuit."""
        return super().label(decimals=decimals, base_label=base_label or "∏_ϕ", cache=cache)

class MultiRZ(Operation):
    r"""
    Arbitrary multi Z rotation.

    .. math::

        MultiRZ(\theta) = \exp\left(-i \frac{\theta}{2} Z^{\otimes n}\right)

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\theta}f(MultiRZ(\theta)) = \frac{1}{2}\left[f(MultiRZ(\theta +\pi/2)) - f(MultiRZ(\theta-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`MultiRZ(\theta)`.

    .. note::

        If the ``MultiRZ`` gate is not supported on the targeted device, PennyLane
        will decompose the gate using :class:`~.RZ` and :class:`~.CNOT` gates.

    Args:
        theta (tensor_like or float): rotation angle :math:`\theta`
        wires (Sequence[int] or int): the wires the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def _flatten(self):
        return self.data, (self.wires, tuple())

    def __init__(self, theta, wires=None, id=None):
        wires = Wires(wires)
        self.hyperparameters["num_wires"] = len(wires)
        super().__init__(theta, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta, num_wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle
            num_wires (int): number of wires the rotation acts on

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.MultiRZ.compute_matrix(torch.tensor(0.1), 2)
        tensor([[0.9988-0.0500j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9988+0.0500j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9988+0.0500j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9988-0.0500j]])
        """
        eigs = qml.math.convert_like(pauli_eigs(num_wires), theta)

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
            eigs = qml.math.cast_like(eigs, 1j)

        if qml.math.ndim(theta) == 0:
            return qml.math.diag(qml.math.exp(-0.5j * theta * eigs))

        diags = qml.math.exp(qml.math.outer(-0.5j * theta, eigs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(
            qml.math.eye(2**num_wires, like=diags), diags
        )

    def generator(self):
        return qml.Hamiltonian([-0.5], [functools.reduce(matmul, [PauliZ(w) for w in self.wires])])

    @staticmethod
    def compute_eigvals(theta, num_wires):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.MultiRZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle
            num_wires (int): number of wires the rotation acts on

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.MultiRZ.compute_eigvals(torch.tensor(0.5), 3)
        tensor([0.9689-0.2474j, 0.9689+0.2474j, 0.9689+0.2474j, 0.9689-0.2474j,
                0.9689+0.2474j, 0.9689-0.2474j, 0.9689-0.2474j, 0.9689+0.2474j])
        """
        eigs = qml.math.convert_like(pauli_eigs(num_wires), theta)

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
            eigs = qml.math.cast_like(eigs, 1j)

        if qml.math.ndim(theta) == 0:
            return qml.math.exp(-0.5j * theta * eigs)

        return qml.math.exp(qml.math.outer(-0.5j * theta, eigs))

    @staticmethod
    def compute_decomposition(
        theta, wires, **kwargs
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.MultiRZ.decomposition`.

        Args:
            theta (float): rotation angle :math:`\theta`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.MultiRZ.compute_decomposition(1.2, wires=(0,1))
        [CNOT(wires=[1, 0]), RZ(1.2, wires=[0]), CNOT(wires=[1, 0])]

        """
        ops = [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[~0:0:-1], wires[~1::-1])]
        ops.append(RZ(theta, wires=wires[0]))
        ops += [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[1:], wires[:~0])]

        return ops

    def adjoint(self):
        return MultiRZ(-self.parameters[0], wires=self.wires)

    def pow(self, z):
        return [MultiRZ(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires[0])

        return MultiRZ(theta, wires=self.wires)

