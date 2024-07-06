def _to_string(fermi_op, of=False):
    r"""Return a string representation of the :class:`~.FermiWord` object.

    Args:
        fermi_op (FermiWord): the fermionic operator
        of (bool): whether to return a string representation in the same style as OpenFermion using
                    the shorthand: 'q^' = a^\dagger_q 'q' = a_q. Each operator in the word is
                    represented by the number of the wire it operates on

    Returns:
        (str): a string representation of the :class:`~.FermiWord` object

    **Example**

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> _to_string(w)
    '0+ 1-'

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> _to_string(w, of=True)
    '0^ 1'
    """
    if not isinstance(fermi_op, FermiWord):
        raise ValueError(f"fermi_op must be a FermiWord, got: {type(fermi_op)}")

    pl_to_of_map = {"+": "^", "-": ""}

    if len(fermi_op) == 0:
        return "I"

    op_list = ["" for _ in range(len(fermi_op))]
    for loc, wire in fermi_op:
        if of:
            op_str = str(wire) + pl_to_of_map[fermi_op[(loc, wire)]]
        else:
            op_str = str(wire) + fermi_op[(loc, wire)]

        op_list[loc] += op_str

    return " ".join(op_list).rstrip()

def from_string(fermi_string):
    r"""Return a fermionic operator object from its string representation.

    The string representation is a compact format that uses the orbital index and ``'+'`` or ``'-'``
    symbols to indicate creation and annihilation operators, respectively. For instance, the string
    representation for the operator :math:`a^{\dagger}_0 a_1 a^{\dagger}_0 a_1` is
    ``'0+ 1- 0+ 1-'``. The ``'-'`` symbols can be optionally dropped such that ``'0+ 1 0+ 1'``
    represents the same operator. The format commonly used in OpenFermion to represent the same
    operator, ``'0^ 1 0^ 1'`` , is also supported.

    Args:
        fermi_string (str): string representation of the fermionic object

    Returns:
        FermiWord: the fermionic operator object

    **Example**

    >>> from_string('0+ 1- 0+ 1-')
    a⁺(0) a(1) a⁺(0) a(1)

    >>> from_string('0+ 1 0+ 1')
    a⁺(0) a(1) a⁺(0) a(1)

    >>> from_string('0^ 1 0^ 1')
    a⁺(0) a(1) a⁺(0) a(1)

    >>> op1 = FermiC(0) * FermiA(1) * FermiC(2) * FermiA(3)
    >>> op2 = from_string('0+ 1- 2+ 3-')
    >>> op1 == op2
    True
    """
    if fermi_string.isspace() or not fermi_string:
        return FermiWord({})

    fermi_string = " ".join(fermi_string.split())

    if not all(s.isdigit() or s in ["+", "-", "^", " "] for s in fermi_string):
        raise ValueError(f"Invalid character encountered in string {fermi_string}.")

    fermi_string = re.sub(r"\^", "+", fermi_string)

    operators = [i + "-" if i[-1] not in "+-" else i for i in re.split(r"\s", fermi_string)]

    return FermiWord({(i, int(s[:-1])): s[-1] for i, s in enumerate(operators)})

