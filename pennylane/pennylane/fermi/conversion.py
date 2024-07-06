def bravyi_kitaev(
    fermi_operator: Union[FermiWord, FermiSentence],
    n: int,
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
) -> Union[Operator, PauliSentence]:
    r"""Convert a fermionic operator to a qubit operator using the Bravyi-Kitaev mapping.

    .. note::

        Hamiltonians created with this mapping should be used with operators and states that are
        compatible with the Bravyi-Kitaev basis.

    In the Bravyi-Kitaev mapping, both occupation number and parity of the orbitals are stored non-locally.
    In comparison, :func:`~.jordan_wigner` stores the occupation number locally while storing the parity
    non-locally and vice-versa for :func:`~.parity_transform`. In the Bravyi-Kitaev mapping, the
    fermionic creation and annihilation operators for even-labelled orbitals are mapped to the Pauli operators as

    .. math::

        \begin{align*}
           a^{\dagger}_0 &= \frac{1}{2} \left ( X_0  -iY_{0} \right ), \\\\
           a^{\dagger}_n &= \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} -iX_{U(n)} \otimes Y_{n} \otimes Z_{P(n)}\right ), \\\\
        \end{align*}

    and

    .. math::
        \begin{align*}
           a_0 &= \frac{1}{2} \left ( X_0  + iY_{0} \right ), \\\\
           a_n &= \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} +iX_{U(n)} \otimes Y_{n} \otimes Z_{P(n)}\right ). \\\\
        \end{align*}

    Similarly, the fermionic creation and annihilation operators for odd-labelled orbitals are mapped to the Pauli operators as

    .. math::
        \begin{align*}
           a^{\dagger}_n &= \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} -iX_{U(n)} \otimes Y_{n} \otimes Z_{R(n)}\right ), \\\\
        \end{align*}

    and

    .. math::
        \begin{align*}
           a_n &= \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} +iX_{U(n)} \otimes Y_{n} \otimes Z_{R(n)}\right ), \\\\
        \end{align*}

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators, and :math:`U(n)`, :math:`P(n)` and :math:`R(n)`
    represent the update, parity and remainder sets, respectively [`arXiv:1812.02233 <https://arxiv.org/abs/1812.02233>`_].

    Args:
        fermi_operator(FermiWord, FermiSentence): the fermionic operator
        n (int): number of qubits, i.e., spin orbitals in the system
        ps (bool): whether to return the result as a PauliSentence instead of an
            Operator. Defaults to False.
        wire_map (dict): a dictionary defining how to map the orbitals of
            the Fermi operator to qubit wires. If None, the integers used to
            order the orbitals will be used as wire labels. Defaults to None.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = qml.fermi.from_string('0+ 1-')
    >>> bravyi_kitaev(w, n=6)
    (
        -0.25j * Y(0)
      + (-0.25+0j) * (X(0) @ Z(1))
      + (0.25+0j) * X(0)
      + 0.25j * (Y(0) @ Z(1))
    )

    >>> bravyi_kitaev(w, n=6, ps=True)
    -0.25j * Y(0)
    + (-0.25+0j) * X(0) @ Z(1)
    + (0.25+0j) * X(0)
    + 0.25j * Y(0) @ Z(1)

    >>> bravyi_kitaev(w, n=6, ps=True, wire_map={0: 2, 1: 3})
    -0.25j * Y(2)
    + (-0.25+0j) * X(2) @ Z(3)
    + (0.25+0j) * X(2)
    + 0.25j * Y(2) @ Z(3)

    """
    return _bravyi_kitaev_dispatch(fermi_operator, n, ps, wire_map, tol)

def parity_transform(
    fermi_operator: Union[FermiWord, FermiSentence],
    n: int,
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
) -> Union[Operator, PauliSentence]:
    r"""Convert a fermionic operator to a qubit operator using the parity mapping.

    .. note::

        Hamiltonians created with this mapping should be used with operators and states that are
        compatible with the parity basis.

    In parity mapping, qubit :math:`j` stores the parity of all :math:`j-1` qubits before it.
    In comparison, :func:`~.jordan_wigner` simply uses qubit :math:`j` to store the occupation number.
    In parity mapping, the fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::
        \begin{align*}
           a^{\dagger}_0 &= \left (\frac{X_0 - iY_0}{2}  \right )\otimes X_1 \otimes X_2 \otimes ... X_n, \\\\
           a^{\dagger}_n &= \left (\frac{Z_{n-1} \otimes X_n - iY_n}{2} \right ) \otimes X_{n+1} \otimes X_{n+2} \otimes ... \otimes X_n
        \end{align*}

    and

    .. math::
        \begin{align*}
           a_0 &= \left (\frac{X_0 + iY_0}{2}  \right )\otimes X_1 \otimes X_2 \otimes ... X_n,\\\\
           a_n &= \left (\frac{Z_{n-1} \otimes X_n + iY_n}{2} \right ) \otimes X_{n+1} \otimes X_{n+2} \otimes ... \otimes X_n
        \end{align*}

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators and :math:`n` is the number of qubits, i.e., spin orbitals.

    Args:
        fermi_operator(FermiWord, FermiSentence): the fermionic operator
        n (int): number of qubits, i.e., spin orbitals in the system
        ps (bool): whether to return the result as a :class:`~.PauliSentence` instead of an
            :class:`~.Operator`. Defaults to ``False``.
        wire_map (dict): a dictionary defining how to map the orbitals of
            the Fermi operator to qubit wires. If ``None``, the integers used to
            order the orbitals will be used as wire labels. Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = qml.fermi.from_string('0+ 1-')
    >>> parity_transform(w, n=6)
    (
        -0.25j * Y(0)
      + (-0.25+0j) * (X(0) @ Z(1))
      + (0.25+0j) * X(0)
      + 0.25j * (Y(0) @ Z(1))
    )

    >>> parity_transform(w, n=6, ps=True)
    -0.25j * Y(0)
    + (-0.25+0j) * X(0) @ Z(1)
    + (0.25+0j) * X(0)
    + 0.25j * Y(0) @ Z(1)

    >>> parity_transform(w, n=6, ps=True, wire_map={0: 2, 1: 3})
    -0.25j * Y(2)
    + (-0.25+0j) * X(2) @ Z(3)
    + (0.25+0j) * X(2)
    + 0.25j * Y(2) @ Z(3)
    """

    return _parity_transform_dispatch(fermi_operator, n, ps, wire_map, tol)

def jordan_wigner(
    fermi_operator: Union[FermiWord, FermiSentence],
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
) -> Union[Operator, PauliSentence]:
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    The fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        a^{\dagger}_0 =  \left (\frac{X_0 - iY_0}{2}  \right ), \:\: \text{...,} \:\:
        a^{\dagger}_n = Z_0 \otimes Z_1 \otimes ... \otimes Z_{n-1} \otimes \left (\frac{X_n - iY_n}{2} \right ),

    and

    .. math::

        a_0 =  \left (\frac{X_0 + iY_0}{2}  \right ), \:\: \text{...,} \:\:
        a_n = Z_0 \otimes Z_1 \otimes ... \otimes Z_{n-1} \otimes \left (\frac{X_n + iY_n}{2}  \right ),

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators.

    Args:
        fermi_operator(FermiWord, FermiSentence): the fermionic operator
        ps (bool): whether to return the result as a PauliSentence instead of an
            Operator. Defaults to False.
        wire_map (dict): a dictionary defining how to map the orbitals of
            the Fermi operator to qubit wires. If None, the integers used to
            order the orbitals will be used as wire labels. Defaults to None.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = qml.fermi.from_string('0+ 1-')
    >>> jordan_wigner(w)
    (
        -0.25j * (Y(0) @ X(1))
      + (0.25+0j) * (Y(0) @ Y(1))
      + (0.25+0j) * (X(0) @ X(1))
      + 0.25j * (X(0) @ Y(1))
    )

    >>> jordan_wigner(w, ps=True)
    -0.25j * Y(0) @ X(1)
    + (0.25+0j) * Y(0) @ Y(1)
    + (0.25+0j) * X(0) @ X(1)
    + 0.25j * X(0) @ Y(1)

    >>> jordan_wigner(w, ps=True, wire_map={0: 2, 1: 3})
    -0.25j * Y(2) @ X(3)
    + (0.25+0j) * Y(2) @ Y(3)
    + (0.25+0j) * X(2) @ X(3)
    + 0.25j * X(2) @ Y(3)
    """
    return _jordan_wigner_dispatch(fermi_operator, ps, wire_map, tol)

