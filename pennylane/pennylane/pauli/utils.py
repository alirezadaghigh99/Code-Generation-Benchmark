def partition_pauli_group(n_qubits: int) -> List[List[str]]:
    """Partitions the :math:`n`-qubit Pauli group into qubit-wise commuting terms.

    The :math:`n`-qubit Pauli group is composed of :math:`4^{n}` terms that can be partitioned into
    :math:`3^{n}` qubit-wise commuting groups.

    Args:
        n_qubits (int): number of qubits

    Returns:
        List[List[str]]: A collection of qubit-wise commuting groups containing Pauli words as
        strings

    **Example**

    >>> qml.pauli.partition_pauli_group(3)
    [['III', 'IIZ', 'IZI', 'IZZ', 'ZII', 'ZIZ', 'ZZI', 'ZZZ'],
     ['IIX', 'IZX', 'ZIX', 'ZZX'],
     ['IIY', 'IZY', 'ZIY', 'ZZY'],
     ['IXI', 'IXZ', 'ZXI', 'ZXZ'],
     ['IXX', 'ZXX'],
     ['IXY', 'ZXY'],
     ['IYI', 'IYZ', 'ZYI', 'ZYZ'],
     ['IYX', 'ZYX'],
     ['IYY', 'ZYY'],
     ['XII', 'XIZ', 'XZI', 'XZZ'],
     ['XIX', 'XZX'],
     ['XIY', 'XZY'],
     ['XXI', 'XXZ'],
     ['XXX'],
     ['XXY'],
     ['XYI', 'XYZ'],
     ['XYX'],
     ['XYY'],
     ['YII', 'YIZ', 'YZI', 'YZZ'],
     ['YIX', 'YZX'],
     ['YIY', 'YZY'],
     ['YXI', 'YXZ'],
     ['YXX'],
     ['YXY'],
     ['YYI', 'YYZ'],
     ['YYX'],
     ['YYY']]
    """
    # Cover the case where n_qubits may be passed as a float
    if isinstance(n_qubits, float):
        if n_qubits.is_integer():
            n_qubits = int(n_qubits)

    # If not an int, or a float representing a int, raise an error
    if not isinstance(n_qubits, int):
        raise TypeError("Must specify an integer number of qubits.")

    if n_qubits < 0:
        raise ValueError("Number of qubits must be at least 0.")

    if n_qubits == 0:
        return [[""]]

    strings = set()  # tracks all the strings that have already been grouped
    groups = []

    # We know that I and Z always commute on a given qubit. The following generates all product
    # sequences of len(n_qubits) over "FXYZ", with F indicating a free slot that can be swapped for
    # the product over I and Z, and all other terms fixed to the given X/Y/Z. For example, if
    # ``n_qubits = 3`` our first value for ``string`` will be ``('F', 'F', 'F')``. We then expand
    # the product of I and Z over the three free slots, giving
    # ``['III', 'IIZ', 'IZI', 'IZZ', 'ZII', 'ZIZ', 'ZZI', 'ZZZ']``, which is our first group. The
    # next element of ``string`` will be ``('F', 'F', 'X')`` which we use to generate our second
    # group ``['IIX', 'IZX', 'ZIX', 'ZZX']``.
    for string in product("FXYZ", repeat=n_qubits):
        if string not in strings:
            num_free_slots = string.count("F")

            group = []
            commuting = product("IZ", repeat=num_free_slots)

            for commuting_string in commuting:
                commuting_string = list(commuting_string)
                new_string = tuple(commuting_string.pop(0) if s == "F" else s for s in string)

                if new_string not in strings:  # only add if string has not already been grouped
                    group.append("".join(new_string))
                    strings |= {new_string}

            if len(group) > 0:
                groups.append(group)

    return groups

def pauli_word_to_string(pauli_word, wire_map=None):
    """Convert a Pauli word to a string.

    A Pauli word can be either:

    * A single pauli operator (see :class:`~.PauliX` for an example).

    * A :class:`.Tensor` instance containing Pauli operators.

    * A :class:`.Prod` instance containing Pauli operators.

    * A :class:`.SProd` instance containing a Pauli operator.

    * A :class:`.Hamiltonian` instance with only one term.

    Given a Pauli in observable form, convert it into string of
    characters from ``['I', 'X', 'Y', 'Z']``. This representation is required for
    functions such as :class:`.PauliRot`.

    .. warning::

        This method ignores any potential coefficient multiplying the Pauli word:

        >>> qml.pauli.pauli_word_to_string(3 * qml.X(0) @ qml.Y(1))
        'XY'

    .. warning::

        This method assumes all Pauli operators are acting on different wires, ignoring
        any extra operators:

        >>> qml.pauli.pauli_word_to_string(qml.X(0) @ qml.Y(0) @ qml.Y(0))
        'X'

    Args:
        pauli_word (Observable): an observable, either a :class:`~.Tensor` instance or
            single-qubit observable representing a Pauli group element.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        str: The string representation of the observable in terms of ``'I'``, ``'X'``, ``'Y'``,
        and/or ``'Z'``.

    Raises:
        TypeError: if the input observable is not a proper Pauli word.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> pauli_word = qml.X('a') @ qml.Y('c')
    >>> pauli_word_to_string(pauli_word, wire_map=wire_map)
    'XIY'
    """

    if not is_pauli_word(pauli_word):
        raise TypeError(f"Expected Pauli word observables, instead got {pauli_word}")
    if isinstance(pauli_word, qml.ops.Hamiltonian):
        # hamiltonian contains only one term
        return _pauli_word_to_string_legacy(pauli_word, wire_map)

    pr = next(iter(pauli_word.pauli_rep.keys()))

    # If there is no wire map, we must infer from the structure of Paulis
    if wire_map is None:
        wire_map = {pauli_word.wires.labels[i]: i for i in range(len(pauli_word.wires))}

    n_qubits = len(wire_map)

    # Set default value of all characters to identity
    pauli_string = ["I"] * n_qubits

    for wire, op_label in pr.items():
        pauli_string[wire_map[wire]] = op_label

    return "".join(pauli_string)

def pauli_to_binary(pauli_word, n_qubits=None, wire_map=None, check_is_pauli_word=True):
    # pylint: disable=isinstance-second-argument-not-valid-type
    """Converts a Pauli word to the binary vector representation.

    This functions follows convention that the first half of binary vector components specify
    PauliX placements while the last half specify PauliZ placements.

    Args:
        pauli_word (Union[Identity, PauliX, PauliY, PauliZ, Tensor, Prod, SProd]): the Pauli word to be
            converted to binary vector representation
        n_qubits (int): number of qubits to specify dimension of binary vector representation
        wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys, and
            unique integer labels as their values
        check_is_pauli_word (bool): If True (default) then a check is run to verify that pauli_word
            is infact a Pauli word

    Returns:
        array: the ``2*n_qubits`` dimensional binary vector representation of the input Pauli word

    Raises:
        TypeError: if the input ``pauli_word`` is not an instance of Identity, PauliX, PauliY,
            PauliZ or tensor products thereof
        ValueError: if ``n_qubits`` is less than the number of wires acted on by the Pauli word

    **Example**

    If ``n_qubits`` and ``wire_map`` are both unspecified, the dimensionality of the binary vector
    will be ``2 * len(pauli_word.wires)``. Regardless of wire labels, the vector components encoding
    Pauli operations will be read from left-to-right in the tensor product when ``wire_map`` is
    unspecified, e.g.,

    >>> pauli_to_binary(qml.X('a') @ qml.Y('b') @ qml.Z('c'))
    array([1., 1., 0., 0., 1., 1.])
    >>> pauli_to_binary(qml.X('c') @ qml.Y('a') @ qml.Z('b'))
    array([1., 1., 0., 0., 1., 1.])

    The above cases have the same binary representation since they are equivalent up to a
    relabelling of the wires. To keep binary vector component enumeration consistent with wire
    labelling across multiple Pauli words, or define any arbitrary enumeration, one can use
    keyword argument ``wire_map`` to set this enumeration.

    >>> wire_map = {'a': 0, 'b': 1, 'c': 2}
    >>> pauli_to_binary(qml.X('a') @ qml.Y('b') @ qml.Z('c'), wire_map=wire_map)
    array([1., 1., 0., 0., 1., 1.])
    >>> pauli_to_binary(qml.X('c') @ qml.Y('a') @ qml.Z('b'), wire_map=wire_map)
    array([1., 0., 1., 1., 1., 0.])

    Now the two Pauli words are distinct in the binary vector representation, as the vector
    components are consistently mapped from the wire labels, rather than enumerated
    left-to-right.

    If ``n_qubits`` is unspecified, the dimensionality of the vector representation will be inferred
    from the size of support of the Pauli word,

    >>> pauli_to_binary(qml.X(0) @ qml.X(1))
    array([1., 1., 0., 0.])
    >>> pauli_to_binary(qml.X(0) @ qml.X(5))
    array([1., 1., 0., 0.])

    Dimensionality higher than twice the support can be specified by ``n_qubits``,

    >>> pauli_to_binary(qml.X(0) @ qml.X(1), n_qubits=6)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> pauli_to_binary(qml.X(0) @ qml.X(5), n_qubits=6)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    For these Pauli words to have a consistent mapping to vector representation, we once again
    need to specify a ``wire_map``.

    >>> wire_map = {0:0, 1:1, 5:5}
    >>> pauli_to_binary(qml.X(0) @ qml.X(1), n_qubits=6, wire_map=wire_map)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> pauli_to_binary(qml.X(0) @ qml.X(5), n_qubits=6, wire_map=wire_map)
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    Note that if ``n_qubits`` is unspecified and ``wire_map`` is specified, the dimensionality of the
    vector representation will be inferred from the highest integer in ``wire_map.values()``.

    >>> wire_map = {0:0, 1:1, 5:5}
    >>> pauli_to_binary(qml.X(0) @ qml.X(5),  wire_map=wire_map)
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    """
    wire_map = wire_map or {w: i for i, w in enumerate(pauli_word.wires)}

    if check_is_pauli_word and not is_pauli_word(pauli_word):
        raise TypeError(f"Expected a Pauli word Observable instance, instead got {pauli_word}.")

    pw = next(iter(pauli_word.pauli_rep))

    n_qubits_min = max(wire_map.values()) + 1
    if n_qubits is None:
        n_qubits = n_qubits_min
    elif n_qubits < n_qubits_min:
        raise ValueError(
            f"n_qubits must support the highest mapped wire index {n_qubits_min},"
            f" instead got n_qubits={n_qubits}."
        )

    binary_pauli = np.zeros(2 * n_qubits)

    for wire, pauli_type in pw.items():
        if pauli_type == "X":
            binary_pauli[wire_map[wire]] = 1
        elif pauli_type == "Y":
            binary_pauli[wire_map[wire]] = 1
            binary_pauli[n_qubits + wire_map[wire]] = 1
        elif pauli_type == "Z":
            binary_pauli[n_qubits + wire_map[wire]] = 1
    return binary_pauli

def is_pauli_word(observable):
    """
    Checks if an observable instance consists only of Pauli and Identity Operators.

    A Pauli word can be either:

    * A single pauli operator (see :class:`~.PauliX` for an example).

    * A :class:`.Tensor` instance containing Pauli operators.

    * A :class:`.Prod` instance containing Pauli operators.

    * A :class:`.SProd` instance containing a valid Pauli word.

    * A :class:`.Hamiltonian` instance with only one term.

    .. Warning::

        This function will only confirm that all operators are Pauli or Identity operators,
        and not whether the Observable is mathematically a Pauli word.
        If an Observable consists of multiple Pauli operators targeting the same wire, the
        function will return ``True`` regardless of any complex coefficients.


    Args:
        observable (~.Operator): the operator to be examined

    Returns:
        bool: true if the input observable is a Pauli word, false otherwise.

    **Example**

    >>> is_pauli_word(qml.Identity(0))
    True
    >>> is_pauli_word(qml.X(0) @ qml.Z(2))
    True
    >>> is_pauli_word(qml.Z(0) @ qml.Hadamard(1))
    False
    >>> is_pauli_word(4 * qml.X(0) @ qml.Z(0))
    True
    """
    return _is_pauli_word(observable) or (len(observable.pauli_rep or []) == 1)

def qwc_complement_adj_matrix(binary_observables):
    """Obtains the adjacency matrix for the complementary graph of the qubit-wise commutativity
    graph for a given set of observables in the binary representation.

    The qubit-wise commutativity graph for a set of Pauli words has a vertex for each Pauli word,
    and two nodes are connected if and only if the corresponding Pauli words are qubit-wise
    commuting.

    Args:
        binary_observables (array[array[int]]): a matrix whose rows are the Pauli words in the
            binary vector representation

    Returns:
        array[array[int]]: the adjacency matrix for the complement of the qubit-wise commutativity graph

    Raises:
        ValueError: if input binary observables contain components which are not strictly binary

    **Example**

    >>> binary_observables
    array([[1., 0., 1., 0., 0., 1.],
           [0., 1., 1., 1., 0., 1.],
           [0., 0., 0., 1., 0., 0.]])

    >>> qwc_complement_adj_matrix(binary_observables)
    array([[0., 1., 1.],
           [1., 0., 0.],
           [1., 0., 0.]])
    """

    if isinstance(binary_observables, (list, tuple)):
        binary_observables = np.asarray(binary_observables)

    if not np.array_equal(binary_observables, binary_observables.astype(bool)):
        raise ValueError(f"Expected a binary array, instead got {binary_observables}")

    m_terms = np.shape(binary_observables)[0]
    adj = np.zeros((m_terms, m_terms))

    for i in range(m_terms):
        for j in range(i + 1, m_terms):
            adj[i, j] = int(not is_qwc(binary_observables[i], binary_observables[j]))
            adj[j, i] = adj[i, j]

    return adj

