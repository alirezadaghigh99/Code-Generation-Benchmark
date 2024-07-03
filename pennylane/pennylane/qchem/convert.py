def import_operator(qubit_observable, format="openfermion", wires=None, tol=1e010):
    r"""Convert an external operator to a PennyLane operator.

    We currently support `OpenFermion <https://quantumai.google/openfermion>`__ operators: the function accepts most types of
    OpenFermion qubit operators, such as those corresponding to Pauli words and sums of Pauli words.

    Args:
        qubit_observable: external qubit operator that will be converted
        format (str): the format of the operator object to convert from
        wires (.Wires, list, tuple, dict): Custom wire mapping used to convert the external qubit
            operator to a PennyLane operator.
            For types ``Wires``/list/tuple, each item in the iterable represents a wire label
            for the corresponding qubit index.
            For type dict, only int-keyed dictionaries (for qubit-to-wire conversion) are accepted.
            If ``None``, the identity map (e.g., ``0->0, 1->1, ...``) will be used.
        tol (float): Tolerance in `machine epsilon <https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html>`_
            for the imaginary part of the coefficients in ``qubit_observable``.
            Coefficients with imaginary part less than :math:`(2.22 \cdot 10^{-16}) \cdot \text{tol}` are considered to be real.

    Returns:
        (.Operator): PennyLane operator representing any operator expressed as linear combinations of
        Pauli words, e.g.,
        :math:`\sum_{k=0}^{N-1} c_k O_k`

    **Example**

    >>> assert qml.operation.active_new_opmath() == True
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (-0.0548 * X(0 @ X(1) @ Y(2) @ Y(3))) + (0.14297 * Z(0 @ Z(1)))

    If the new op-math is deactivated, a :class:`~Hamiltonian` is returned instead.

    >>> assert qml.operation.active_new_opmath() == False
    >>> from openfermion import QubitOperator
    >>> h_of = QubitOperator('X0 X1 Y2 Y3', -0.0548) + QubitOperator('Z0 Z1', 0.14297)
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (0.14297) [Z0 Z1]
    + (-0.0548) [X0 X1 Y2 Y3]
    """
    if format not in ["openfermion"]:
        raise TypeError(f"Converter does not exist for {format} format.")

    coeffs = np.array([np.real_if_close(coef, tol=tol) for coef in qubit_observable.terms.values()])

    if any(np.iscomplex(coeffs)):
        warnings.warn(
            f"The coefficients entering the QubitOperator must be real;"
            f" got complex coefficients in the operator"
            f" {list(coeffs[np.iscomplex(coeffs)])}"
        )

    if active_new_opmath():
        return qml.dot(*_openfermion_to_pennylane(qubit_observable, wires=wires))

    return qml.Hamiltonian(*_openfermion_to_pennylane(qubit_observable, wires=wires))def _process_wires(wires, n_wires=None):
    r"""Checks and consolidates custom user wire mapping into a consistent, direction-free, ``Wires``
    format. Used for converting between OpenFermion qubit numbering and PennyLane wire labels.

    Since OpenFermion's qubit numbering is always consecutive int, simple iterable types such as
    list, tuple, or Wires can be used to specify the 2-way `qubit index` <-> `wire label` mapping
    with indices representing qubits. Dict can also be used as a mapping, but does not provide any
    advantage over lists other than the ability to do partial mapping/permutation in the
    `qubit index` -> `wire label` direction.

    It is recommended to pass Wires/list/tuple `wires` since it's direction-free, i.e. the same
    `wires` argument can be used to convert both ways between OpenFermion and PennyLane. Only use
    dict for partial or unordered mapping.

    Args:
        wires (Wires, list, tuple, dict): User wire labels.
            For types Wires, list, or tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) or
            consecutive-int-valued dict (for wire-to-qubit conversion) is accepted.
            If None, will be set to consecutive int based on ``n_wires``.
        n_wires (int): Number of wires used if known. If None, will be inferred from ``wires``; if
            ``wires`` is not available, will be set to 1.

    Returns:
        Wires: Cleaned wire mapping with indices corresponding to qubits and values
            corresponding to wire labels.

    **Example**

    >>> # consec int wires if no wires mapping provided, ie. identity map: 0<->0, 1<->1, 2<->2
    >>> _process_wires(None, 3)
    <Wires = [0, 1, 2]>

    >>> # List as mapping, qubit indices with wire label values: 0<->w0, 1<->w1, 2<->w2
    >>> _process_wires(['w0','w1','w2'])
    <Wires = ['w0', 'w1', 'w2']>

    >>> # Wires as mapping, qubit indices with wire label values: 0<->w0, 1<->w1, 2<->w2
    >>> _process_wires(Wires(['w0', 'w1', 'w2']))
    <Wires = ['w0', 'w1', 'w2']>

    >>> # Dict as partial mapping, int qubits keys to wire label values: 0->w0, 1 unchanged, 2->w2
    >>> _process_wires({0:'w0',2:'w2'})
    <Wires = ['w0', 1, 'w2']>

    >>> # Dict as mapping, wires label keys to consec int qubit values: w2->2, w0->0, w1->1
    >>> _process_wires({'w2':2, 'w0':0, 'w1':1})
    <Wires = ['w0', 'w1', 'w2']>
    """

    # infer from wires, or assume 1 if wires is not of accepted types.
    if n_wires is None:
        n_wires = len(wires) if isinstance(wires, (Wires, list, tuple, dict)) else 1

    # defaults to no mapping.
    if wires is None:
        return Wires(range(n_wires))

    if isinstance(wires, (Wires, list, tuple)):
        # does not care about the tail if more wires are provided than n_wires.
        wires = Wires(wires[:n_wires])

    elif isinstance(wires, dict):
        if all(isinstance(w, int) for w in wires.keys()):
            # Assuming keys are taken from consecutive int wires. Allows for partial mapping.
            n_wires = max(wires) + 1
            labels = list(range(n_wires))  # used for completing potential partial mapping.
            for k, v in wires.items():
                if k < n_wires:
                    labels[k] = v
            wires = Wires(labels)
        elif set(range(n_wires)).issubset(set(wires.values())):
            # Assuming values are consecutive int wires (up to n_wires, ignores the rest).
            # Does NOT allow for partial mapping.
            wires = {v: k for k, v in wires.items()}  # flip for easy indexing
            wires = Wires([wires[i] for i in range(n_wires)])
        else:
            raise ValueError("Expected only int-keyed or consecutive int-valued dict for `wires`")

    else:
        raise ValueError(
            f"Expected type Wires, list, tuple, or dict for `wires`, got {type(wires)}"
        )

    if len(wires) != n_wires:
        # check length consistency when all checking and cleaning are done.
        raise ValueError(f"Length of `wires` ({len(wires)}) does not match `n_wires` ({n_wires})")

    return wires