def from_openfermion(openfermion_op, wires=None, tol=1e-16):
    r"""Convert OpenFermion
    `FermionOperator <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`__
    to PennyLane :class:`~.fermi.FermiWord` or :class:`~.fermi.FermiSentence` and
    OpenFermion `QubitOperator <https://quantumai.google/reference/python/openfermion/ops/QubitOperator>`__
    to PennyLane :class:`~.LinearCombination`.

    Args:
        openfermion_op (FermionOperator, QubitOperator): OpenFermion operator
        wires (dict): Custom wire mapping used to convert the external qubit
            operator to a PennyLane operator.
            Only dictionaries with integer keys (for qubit-to-wire conversion) are accepted.
            If ``None``, the identity map (e.g., ``0->0, 1->1, ...``) will be used.
        tol (float): tolerance for discarding negligible coefficients

    Returns:
        Union[FermiWord, FermiSentence, LinearCombination]: PennyLane operator

    **Example**

    >>> from openfermion import FermionOperator, QubitOperator
    >>> of_op = 0.5 * FermionOperator('0^ 2') + FermionOperator('0 2^')
    >>> pl_op = from_openfermion(of_op)
    >>> print(pl_op)
    0.5 * a⁺(0) a(2)
    + 1.0 * a(0) a⁺(2)

    >>> of_op = QubitOperator('X0', 1.2) + QubitOperator('Z1', 2.4)
    >>> pl_op = from_openfermion(of_op)
    >>> print(pl_op)
    1.2 * X(0) + 2.4 * Z(1)
    """
    openfermion = _import_of()

    if isinstance(openfermion_op, openfermion.FermionOperator):

        if wires:
            raise ValueError("Custom wire mapping is not supported for fermionic operators.")

        typemap = {0: "-", 1: "+"}

        fermi_words = []
        fermi_coeffs = []

        for ops, val in openfermion_op.terms.items():
            fw_dict = {(i, op[0]): typemap[op[1]] for i, op in enumerate(ops)}
            fermi_words.append(FermiWord(fw_dict))
            fermi_coeffs.append(val)

        if len(fermi_words) == 1 and fermi_coeffs[0] == 1.0:
            return fermi_words[0]

        pl_op = FermiSentence(dict(zip(fermi_words, fermi_coeffs)))
        pl_op.simplify(tol=tol)

        return pl_op

    coeffs, pl_ops = _openfermion_to_pennylane(openfermion_op, wires=wires, tol=tol)

    pennylane_op = qml.ops.LinearCombination(coeffs, pl_ops)

    return pennylane_op

def to_openfermion(
    pennylane_op: Union[Sum, LinearCombination, FermiWord, FermiSentence], wires=None, tol=1.0e-16
):
    r"""Convert a PennyLane operator to OpenFermion
    `QubitOperator <https://quantumai.google/reference/python/openfermion/ops/QubitOperator>`__ or
    `FermionOperator <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`__.

    Args:
        pennylane_op (~ops.op_math.Sum, ~ops.op_math.LinearCombination, FermiWord, FermiSentence):
            PennyLane operator
        wires (dict): Custom wire mapping used to convert a PennyLane qubit operator
            to the external operator.
            Only dictionaries with integer keys (for qubit-to-wire conversion) are accepted.
            If ``None``, the identity map (e.g., ``0->0, 1->1, ...``) will be used.

    Returns:
        (QubitOperator, FermionOperator): OpenFermion operator

    **Example**

    >>> w1 = qml.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.fermi.FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = qml.fermi.FermiSentence({w1 : 1.2, w2: 3.1})
    >>> of_op = qml.to_openfermion(s)
    >>> of_op
    1.2 [0^ 1] +
    3.1 [1^ 2]
    """

    return _to_openfermion_dispatch(pennylane_op, wires=wires, tol=tol)

def from_openfermion(openfermion_op, wires=None, tol=1e-16):
    r"""Convert OpenFermion
    `FermionOperator <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`__
    to PennyLane :class:`~.fermi.FermiWord` or :class:`~.fermi.FermiSentence` and
    OpenFermion `QubitOperator <https://quantumai.google/reference/python/openfermion/ops/QubitOperator>`__
    to PennyLane :class:`~.LinearCombination`.

    Args:
        openfermion_op (FermionOperator, QubitOperator): OpenFermion operator
        wires (dict): Custom wire mapping used to convert the external qubit
            operator to a PennyLane operator.
            Only dictionaries with integer keys (for qubit-to-wire conversion) are accepted.
            If ``None``, the identity map (e.g., ``0->0, 1->1, ...``) will be used.
        tol (float): tolerance for discarding negligible coefficients

    Returns:
        Union[FermiWord, FermiSentence, LinearCombination]: PennyLane operator

    **Example**

    >>> from openfermion import FermionOperator, QubitOperator
    >>> of_op = 0.5 * FermionOperator('0^ 2') + FermionOperator('0 2^')
    >>> pl_op = from_openfermion(of_op)
    >>> print(pl_op)
    0.5 * a⁺(0) a(2)
    + 1.0 * a(0) a⁺(2)

    >>> of_op = QubitOperator('X0', 1.2) + QubitOperator('Z1', 2.4)
    >>> pl_op = from_openfermion(of_op)
    >>> print(pl_op)
    1.2 * X(0) + 2.4 * Z(1)
    """
    openfermion = _import_of()

    if isinstance(openfermion_op, openfermion.FermionOperator):

        if wires:
            raise ValueError("Custom wire mapping is not supported for fermionic operators.")

        typemap = {0: "-", 1: "+"}

        fermi_words = []
        fermi_coeffs = []

        for ops, val in openfermion_op.terms.items():
            fw_dict = {(i, op[0]): typemap[op[1]] for i, op in enumerate(ops)}
            fermi_words.append(FermiWord(fw_dict))
            fermi_coeffs.append(val)

        if len(fermi_words) == 1 and fermi_coeffs[0] == 1.0:
            return fermi_words[0]

        pl_op = FermiSentence(dict(zip(fermi_words, fermi_coeffs)))
        pl_op.simplify(tol=tol)

        return pl_op

    coeffs, pl_ops = _openfermion_to_pennylane(openfermion_op, wires=wires, tol=tol)

    pennylane_op = qml.ops.LinearCombination(coeffs, pl_ops)

    return pennylane_op

