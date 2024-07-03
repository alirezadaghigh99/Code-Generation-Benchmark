class MultiControlledX(ControlledOp):
    r"""MultiControlledX(control_wires, wires, control_values)
    Apply a Pauli X gate controlled on an arbitrary computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        control_wires (Union[Wires, Sequence[int], or int]): Deprecated way to indicate the control wires.
            Now users should use "wires" to indicate both the control wires and the target wire.
        wires (Union[Wires, Sequence[int], or int]): control wire(s) followed by a single target wire where
            the operation acts on
        control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            the operation into a series of Toffoli gates


    .. note::

        If ``MultiControlledX`` is not supported on the targeted device, PennyLane will decompose
        the operation into :class:`~.Toffoli` and/or :class:`~.CNOT` gates. When controlling on
        three or more wires, the Toffoli-based decompositions described in Lemmas 7.2 and 7.3 of
        `Barenco et al. <https://arxiv.org/abs/quant-ph/9503016>`__ will be used. These methods
        require at least one work wire.

        The number of work wires provided determines the decomposition method used and the resulting
        number of Toffoli gates required. When ``MultiControlledX`` is controlling on :math:`n`
        wires:

        #. If at least :math:`n - 2` work wires are provided, the decomposition in Lemma 7.2 will be
           applied using the first :math:`n - 2` work wires.
        #. If fewer than :math:`n - 2` work wires are provided, a combination of Lemmas 7.3 and 7.2
           will be applied using only the first work wire.

        These methods present a tradeoff between qubit number and depth. The method in point 1
        requires fewer Toffoli gates but a greater number of qubits.

        Note that the state of the work wires before and after the decomposition takes place is
        unchanged.

    """

    is_self_inverse = True
    """bool: Whether or not the operator is self-inverse."""

    num_wires = AnyWires
    """int: Number of wires the operation acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    name = "MultiControlledX"

    def _flatten(self):
        return (), (self.wires, tuple(self.control_values), self.work_wires)

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(wires=metadata[0], control_values=metadata[1], work_wires=metadata[2])

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, wires, control_values=None, work_wires=None, id=None):
        return cls._primitive.bind(
            *wires, n_wires=len(wires), control_values=control_values, work_wires=work_wires
        )

    # pylint: disable=too-many-arguments
    def __init__(self, control_wires=None, wires=None, control_values=None, work_wires=None):

        # First raise deprecation warnings regardless of the validity of other arguments
        if isinstance(control_values, str):
            warnings.warn(
                "Specifying control values using a bitstring is deprecated, and will not be "
                "supported in future releases, Use a list of booleans or integers instead.",
                qml.PennyLaneDeprecationWarning,
            )
        if control_wires is not None:
            warnings.warn(
                "The control_wires keyword for MultiControlledX is deprecated, and will "
                "be removed soon. Use wires = (*control_wires, target_wire) instead.",
                UserWarning,
            )

        if wires is None:
            raise ValueError("Must specify the wires where the operation acts on")

        wires = wires if isinstance(wires, Wires) else Wires(wires)

        if control_wires is not None:
            if len(wires) != 1:
                raise ValueError("MultiControlledX accepts a single target wire.")
            control_wires = Wires(control_wires)
        else:
            if len(wires) < 2:
                raise ValueError(
                    f"MultiControlledX: wrong number of wires. {len(wires)} wire(s) given. "
                    f"Need at least 2."
                )
            control_wires = wires[:-1]
            wires = wires[-1:]

        control_values = _check_and_convert_control_values(control_values, control_wires)

        super().__init__(
            qml.X(wires),
            control_wires=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )

    def __repr__(self):
        return (
            f"MultiControlledX(wires={self.wires.tolist()}, control_values={self.control_values})"
        )

    @property
    def wires(self):
        return self.control_wires + self.target_wires

    # pylint: disable=unused-argument, arguments-differ
    @staticmethod
    def compute_matrix(control_wires, control_values=None, **kwargs):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiControlledX.matrix`

        Args:
            control_wires (Any or Iterable[Any]): wires to place controls on
            control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            tensor_like: matrix representation

        **Example**

        >>> print(qml.MultiControlledX.compute_matrix([0], [1]))
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]
         [0. 0. 1. 0.]]
        >>> print(qml.MultiControlledX.compute_matrix([1], [0]))
        [[0. 1. 0. 0.]
         [1. 0. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]]

        """

        control_values = _check_and_convert_control_values(control_values, control_wires)
        padding_left = sum(2**i * int(val) for i, val in enumerate(reversed(control_values))) * 2
        padding_right = 2 ** (len(control_wires) + 1) - 2 - padding_left
        return block_diag(np.eye(padding_left), qml.X.compute_matrix(), np.eye(padding_right))

    def matrix(self, wire_order=None):
        canonical_matrix = self.compute_matrix(self.control_wires, self.control_values)
        wire_order = wire_order or self.wires
        return qml.math.expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    # pylint: disable=unused-argument, arguments-differ
    @staticmethod
    def compute_decomposition(wires=None, work_wires=None, control_values=None, **kwargs):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.MultiControlledX.decomposition`.

        Args:
            wires (Iterable[Any] or Wires): wires that the operation acts on
            work_wires (Wires): optional work wires used to decompose
                the operation into a series of Toffoli gates.
            control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.MultiControlledX.compute_decomposition(
        ...     wires=[0,1,2,3], control_values=[1,1,1], work_wires=qml.wires.Wires("aux")))
        [Toffoli(wires=[2, 'aux', 3]),
        Toffoli(wires=[0, 1, 'aux']),
        Toffoli(wires=[2, 'aux', 3]),
        Toffoli(wires=[0, 1, 'aux'])]

        """

        if len(wires) < 2:
            raise ValueError(f"Wrong number of wires. {len(wires)} given. Need at least 2.")

        target_wire = wires[-1]
        control_wires = wires[:-1]

        if control_values is None:
            control_values = [True] * len(control_wires)

        work_wires = work_wires or []

        flips1 = [qml.X(w) for w, val in zip(control_wires, control_values) if not val]

        decomp = decompose_mcx(control_wires, target_wire, work_wires)

        flips2 = [qml.X(w) for w, val in zip(control_wires, control_values) if not val]

        return flips1 + decomp + flips2

    def decomposition(self):
        return self.compute_decomposition(self.wires, self.work_wires, self.control_values)class ControlledQubitUnitary(ControlledOp):
    r"""ControlledQubitUnitary(U, control_wires, wires, control_values)
    Apply an arbitrary fixed unitary to ``wires`` with control from the ``control_wires``.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQubitUnitary``:

    * ``control_wires``: wires that act as control for the operation
    * ``control_values``: the state on which to apply the controlled operation (see below)
    * ``target_wires``: the wires the unitary matrix will be applied to
    * ``work_wires``: wires made use of during the decomposition of the operation into native operations

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        base (Union[array[complex], QubitUnitary]): square unitary matrix or a QubitUnitary
            operation. If passing a matrix, this will be used to construct a QubitUnitary
            operator that will be used as the base operator. If providing a ``qml.QubitUnitary``,
            this will be used as the base directly.
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the unitary acts on
            (optional if U is provided as a QubitUnitary)
        control_values (List[int, bool]): a list providing the state of the control qubits to
            control on (default is the all 1s state)
        unitary_check (bool): whether to check whether an array U is unitary when creating the
            operator (default False)
        work_wires (Union[Wires, Sequence[int], or int]): ancillary wire(s) that may be utilized in during
            the decomposition of the operator into native operations.

    **Example**

    The following shows how a single-qubit unitary can be applied to wire ``2`` with control on
    both wires ``0`` and ``1``:

    >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1], wires=2)
    Controlled(QubitUnitary(array([[ 0.94877869,  0.31594146],
       [-0.31594146,  0.94877869]]), wires=[2]), control_wires=[0, 1])

    Alternatively, the same operator can be constructed with a QubitUnitary:

    >>> base = qml.QubitUnitary(U, wires=2)
    >>> qml.ControlledQubitUnitary(base, control_wires=[0, 1])
    Controlled(QubitUnitary(array([[ 0.94877869,  0.31594146],
       [-0.31594146,  0.94877869]]), wires=[2]), control_wires=[0, 1])

    Typically, controlled operations apply a desired gate if the control qubits
    are all in the state :math:`\vert 1\rangle`. However, there are some situations where
    it is necessary to apply a gate conditioned on all qubits being in the
    :math:`\vert 0\rangle` state, or a mix of the two.

    The state on which to control can be changed by passing a string of bits to
    `control_values`. For example, if we want to apply a single-qubit unitary to
    wire ``3`` conditioned on three wires where the first is in state ``0``, the
    second is in state ``1``, and the third in state ``1``, we can write:

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values=[0, 1, 1])

    or

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values=[False, True, True])
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(
            data[0], control_wires=metadata[0], control_values=metadata[1], work_wires=metadata[2]
        )

    # pylint: disable= too-many-arguments
    def __init__(
        self,
        base,
        control_wires,
        wires=None,
        control_values=None,
        unitary_check=False,
        work_wires=None,
    ):
        if getattr(base, "wires", False) and wires is not None:
            warnings.warn(
                "base operator already has wires; values specified through wires kwarg will be ignored."
            )

        if isinstance(base, Iterable):
            base = QubitUnitary(base, wires=wires, unitary_check=unitary_check)

        super().__init__(
            base,
            control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )
        self._name = "ControlledQubitUnitary"

    def _controlled(self, wire):
        ctrl_wires = wire + self.control_wires
        values = None if self.control_values is None else [True] + self.control_values
        return ControlledQubitUnitary(
            self.base,
            control_wires=ctrl_wires,
            control_values=values,
            work_wires=self.work_wires,
        )

    @property
    def has_decomposition(self):
        if not super().has_decomposition:
            return False
        with qml.QueuingManager.stop_recording():
            # we know this is using try-except as logical control, but are favouring
            # certainty in it being correct over explicitness in an edge case.
            try:
                self.decomposition()
            except qml.operation.DecompositionUndefinedError:
                return False
        return Trueclass MultiControlledX(ControlledOp):
    r"""MultiControlledX(control_wires, wires, control_values)
    Apply a Pauli X gate controlled on an arbitrary computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        control_wires (Union[Wires, Sequence[int], or int]): Deprecated way to indicate the control wires.
            Now users should use "wires" to indicate both the control wires and the target wire.
        wires (Union[Wires, Sequence[int], or int]): control wire(s) followed by a single target wire where
            the operation acts on
        control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            the operation into a series of Toffoli gates


    .. note::

        If ``MultiControlledX`` is not supported on the targeted device, PennyLane will decompose
        the operation into :class:`~.Toffoli` and/or :class:`~.CNOT` gates. When controlling on
        three or more wires, the Toffoli-based decompositions described in Lemmas 7.2 and 7.3 of
        `Barenco et al. <https://arxiv.org/abs/quant-ph/9503016>`__ will be used. These methods
        require at least one work wire.

        The number of work wires provided determines the decomposition method used and the resulting
        number of Toffoli gates required. When ``MultiControlledX`` is controlling on :math:`n`
        wires:

        #. If at least :math:`n - 2` work wires are provided, the decomposition in Lemma 7.2 will be
           applied using the first :math:`n - 2` work wires.
        #. If fewer than :math:`n - 2` work wires are provided, a combination of Lemmas 7.3 and 7.2
           will be applied using only the first work wire.

        These methods present a tradeoff between qubit number and depth. The method in point 1
        requires fewer Toffoli gates but a greater number of qubits.

        Note that the state of the work wires before and after the decomposition takes place is
        unchanged.

    """

    is_self_inverse = True
    """bool: Whether or not the operator is self-inverse."""

    num_wires = AnyWires
    """int: Number of wires the operation acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    name = "MultiControlledX"

    def _flatten(self):
        return (), (self.wires, tuple(self.control_values), self.work_wires)

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(wires=metadata[0], control_values=metadata[1], work_wires=metadata[2])

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, wires, control_values=None, work_wires=None, id=None):
        return cls._primitive.bind(
            *wires, n_wires=len(wires), control_values=control_values, work_wires=work_wires
        )

    # pylint: disable=too-many-arguments
    def __init__(self, control_wires=None, wires=None, control_values=None, work_wires=None):

        # First raise deprecation warnings regardless of the validity of other arguments
        if isinstance(control_values, str):
            warnings.warn(
                "Specifying control values using a bitstring is deprecated, and will not be "
                "supported in future releases, Use a list of booleans or integers instead.",
                qml.PennyLaneDeprecationWarning,
            )
        if control_wires is not None:
            warnings.warn(
                "The control_wires keyword for MultiControlledX is deprecated, and will "
                "be removed soon. Use wires = (*control_wires, target_wire) instead.",
                UserWarning,
            )

        if wires is None:
            raise ValueError("Must specify the wires where the operation acts on")

        wires = wires if isinstance(wires, Wires) else Wires(wires)

        if control_wires is not None:
            if len(wires) != 1:
                raise ValueError("MultiControlledX accepts a single target wire.")
            control_wires = Wires(control_wires)
        else:
            if len(wires) < 2:
                raise ValueError(
                    f"MultiControlledX: wrong number of wires. {len(wires)} wire(s) given. "
                    f"Need at least 2."
                )
            control_wires = wires[:-1]
            wires = wires[-1:]

        control_values = _check_and_convert_control_values(control_values, control_wires)

        super().__init__(
            qml.X(wires),
            control_wires=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )

    def __repr__(self):
        return (
            f"MultiControlledX(wires={self.wires.tolist()}, control_values={self.control_values})"
        )

    @property
    def wires(self):
        return self.control_wires + self.target_wires

    # pylint: disable=unused-argument, arguments-differ
    @staticmethod
    def compute_matrix(control_wires, control_values=None, **kwargs):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiControlledX.matrix`

        Args:
            control_wires (Any or Iterable[Any]): wires to place controls on
            control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            tensor_like: matrix representation

        **Example**

        >>> print(qml.MultiControlledX.compute_matrix([0], [1]))
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]
         [0. 0. 1. 0.]]
        >>> print(qml.MultiControlledX.compute_matrix([1], [0]))
        [[0. 1. 0. 0.]
         [1. 0. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]]

        """

        control_values = _check_and_convert_control_values(control_values, control_wires)
        padding_left = sum(2**i * int(val) for i, val in enumerate(reversed(control_values))) * 2
        padding_right = 2 ** (len(control_wires) + 1) - 2 - padding_left
        return block_diag(np.eye(padding_left), qml.X.compute_matrix(), np.eye(padding_right))

    def matrix(self, wire_order=None):
        canonical_matrix = self.compute_matrix(self.control_wires, self.control_values)
        wire_order = wire_order or self.wires
        return qml.math.expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    # pylint: disable=unused-argument, arguments-differ
    @staticmethod
    def compute_decomposition(wires=None, work_wires=None, control_values=None, **kwargs):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.MultiControlledX.decomposition`.

        Args:
            wires (Iterable[Any] or Wires): wires that the operation acts on
            work_wires (Wires): optional work wires used to decompose
                the operation into a series of Toffoli gates.
            control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.MultiControlledX.compute_decomposition(
        ...     wires=[0,1,2,3], control_values=[1,1,1], work_wires=qml.wires.Wires("aux")))
        [Toffoli(wires=[2, 'aux', 3]),
        Toffoli(wires=[0, 1, 'aux']),
        Toffoli(wires=[2, 'aux', 3]),
        Toffoli(wires=[0, 1, 'aux'])]

        """

        if len(wires) < 2:
            raise ValueError(f"Wrong number of wires. {len(wires)} given. Need at least 2.")

        target_wire = wires[-1]
        control_wires = wires[:-1]

        if control_values is None:
            control_values = [True] * len(control_wires)

        work_wires = work_wires or []

        flips1 = [qml.X(w) for w, val in zip(control_wires, control_values) if not val]

        decomp = decompose_mcx(control_wires, target_wire, work_wires)

        flips2 = [qml.X(w) for w, val in zip(control_wires, control_values) if not val]

        return flips1 + decomp + flips2

    def decomposition(self):
        return self.compute_decomposition(self.wires, self.work_wires, self.control_values)