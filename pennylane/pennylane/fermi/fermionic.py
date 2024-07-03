class FermiSentence(dict):
    r"""Immutable dictionary used to represent a Fermi sentence, a linear combination of Fermi words, with the keys
    as FermiWord instances and the values correspond to coefficients.

    >>> w1 = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = FermiSentence({w1 : 1.2, w2: 3.1})
    >>> s
    1.2 * a⁺(0) a(1)
    + 3.1 * a⁺(1) a(2)
    """

    # override the arithmetic dunder methods for numpy arrays so that the
    # methods defined on this class are used instead
    # (i.e. ensure `np.array + FermiSentence` uses `FermiSentence.__radd__` instead of `np.array.__add__`)
    __numpy_ufunc__ = None
    __array_ufunc__ = None

    def __init__(self, operator):
        super().__init__(operator)

    @property
    def wires(self):
        r"""Return wires of the FermiSentence."""
        return set().union(*(fw.wires for fw in self.keys()))

    def __str__(self):
        r"""String representation of a FermiSentence."""
        if len(self) == 0:
            return "0 * I"
        return "\n+ ".join(f"{coeff} * {fw.to_string()}" for fw, coeff in self.items())

    def __repr__(self):
        r"""Terminal representation for FermiSentence."""
        return str(self)

    def __missing__(self, key):
        r"""If the FermiSentence does not contain a FermiWord then the associated value will be 0."""
        return 0.0

    def __add__(self, other):
        r"""Add a FermiSentence, FermiWord or constant to a FermiSentence by iterating over the
        smaller one and adding its terms to the larger one."""

        # ensure other is FermiSentence
        if isinstance(other, FermiWord):
            other = FermiSentence({other: 1})
        if isinstance(other, Number):
            other = FermiSentence({FermiWord({}): other})
        if isinstance(other, ndarray):
            if qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            other = FermiSentence({FermiWord({}): other})

        if isinstance(other, FermiSentence):
            smaller_fs, larger_fs = (
                (self, copy(other)) if len(self) < len(other) else (other, copy(self))
            )
            for key in smaller_fs:
                larger_fs[key] += smaller_fs[key]

            return larger_fs

        raise TypeError(f"Cannot add {type(other)} to a FermiSentence.")

    def __radd__(self, other):
        """Add a FermiSentence to a constant, i.e. `2 + FermiSentence({...})`"""

        if isinstance(other, (Number, ndarray)):
            return self.__add__(other)

        raise TypeError(f"Cannot add a FermiSentence to {type(other)}.")

    def __sub__(self, other):
        r"""Subtract a FermiSentence, FermiWord or constant from a FermiSentence"""
        if isinstance(other, FermiWord):
            other = FermiSentence({other: -1})
            return self.__add__(other)

        if isinstance(other, Number):
            other = FermiSentence({FermiWord({}): -1 * other})  # -constant * I
            return self.__add__(other)

        if isinstance(other, ndarray):
            if qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            other = FermiSentence({FermiWord({}): -1 * other})  # -constant * I
            return self.__add__(other)

        if isinstance(other, FermiSentence):
            other = FermiSentence(dict(zip(other.keys(), [-1 * v for v in other.values()])))
            return self.__add__(other)

        raise TypeError(f"Cannot subtract {type(other)} from a FermiSentence.")

    def __rsub__(self, other):
        """Subtract a FermiSentence to a constant, i.e.

        >>> 2 - FermiSentence({...})
        """

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            self_fs = FermiSentence(dict(zip(self.keys(), [-1 * v for v in self.values()])))
            other_fs = FermiSentence({FermiWord({}): other})  # constant * I
            return self_fs + other_fs

        raise TypeError(f"Cannot subtract a FermiSentence from {type(other)}.")

    def __mul__(self, other):
        r"""Multiply two Fermi sentences by iterating over each sentence and multiplying the Fermi
        words pair-wise"""

        if isinstance(other, FermiWord):
            other = FermiSentence({other: 1})

        if isinstance(other, FermiSentence):
            if (len(self) == 0) or (len(other) == 0):
                return FermiSentence({FermiWord({}): 0})

            product = FermiSentence({})

            for fw1, coeff1 in self.items():
                for fw2, coeff2 in other.items():
                    product[fw1 * fw2] += coeff1 * coeff2

            return product

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            vals = [i * other for i in self.values()]
            return FermiSentence(dict(zip(self.keys(), vals)))

        raise TypeError(f"Cannot multiply FermiSentence by {type(other)}.")

    def __rmul__(self, other):
        r"""Reverse multiply a FermiSentence

        Multiplies a FermiSentence "from the left" with an object that can't be modified
        to support __mul__ for FermiSentence. Will be defaulted in for example when
        multiplying ``2 * fermi_sentence``, since the ``__mul__`` operator on an integer
        will fail to multiply with a FermiSentence"""

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            vals = [i * other for i in self.values()]
            return FermiSentence(dict(zip(self.keys(), vals)))

        raise TypeError(f"Cannot multiply {type(other)} by FermiSentence.")

    def __pow__(self, value):
        r"""Exponentiate a Fermi sentence to an integer power."""
        if value < 0 or not isinstance(value, int):
            raise ValueError("The exponent must be a positive integer.")

        operator = FermiSentence({FermiWord({}): 1})  # 1 times Identity

        for _ in range(value):
            operator *= self

        return operator

    def simplify(self, tol=1e-8):
        r"""Remove any FermiWords in the FermiSentence with coefficients less than the threshold
        tolerance."""
        items = list(self.items())
        for fw, coeff in items:
            if abs(coeff) <= tol:
                del self[fw]def from_string(fermi_string):
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

    return FermiWord({(i, int(s[:-1])): s[-1] for i, s in enumerate(operators)})class FermiSentence(dict):
    r"""Immutable dictionary used to represent a Fermi sentence, a linear combination of Fermi words, with the keys
    as FermiWord instances and the values correspond to coefficients.

    >>> w1 = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = FermiSentence({w1 : 1.2, w2: 3.1})
    >>> s
    1.2 * a⁺(0) a(1)
    + 3.1 * a⁺(1) a(2)
    """

    # override the arithmetic dunder methods for numpy arrays so that the
    # methods defined on this class are used instead
    # (i.e. ensure `np.array + FermiSentence` uses `FermiSentence.__radd__` instead of `np.array.__add__`)
    __numpy_ufunc__ = None
    __array_ufunc__ = None

    def __init__(self, operator):
        super().__init__(operator)

    @property
    def wires(self):
        r"""Return wires of the FermiSentence."""
        return set().union(*(fw.wires for fw in self.keys()))

    def __str__(self):
        r"""String representation of a FermiSentence."""
        if len(self) == 0:
            return "0 * I"
        return "\n+ ".join(f"{coeff} * {fw.to_string()}" for fw, coeff in self.items())

    def __repr__(self):
        r"""Terminal representation for FermiSentence."""
        return str(self)

    def __missing__(self, key):
        r"""If the FermiSentence does not contain a FermiWord then the associated value will be 0."""
        return 0.0

    def __add__(self, other):
        r"""Add a FermiSentence, FermiWord or constant to a FermiSentence by iterating over the
        smaller one and adding its terms to the larger one."""

        # ensure other is FermiSentence
        if isinstance(other, FermiWord):
            other = FermiSentence({other: 1})
        if isinstance(other, Number):
            other = FermiSentence({FermiWord({}): other})
        if isinstance(other, ndarray):
            if qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            other = FermiSentence({FermiWord({}): other})

        if isinstance(other, FermiSentence):
            smaller_fs, larger_fs = (
                (self, copy(other)) if len(self) < len(other) else (other, copy(self))
            )
            for key in smaller_fs:
                larger_fs[key] += smaller_fs[key]

            return larger_fs

        raise TypeError(f"Cannot add {type(other)} to a FermiSentence.")

    def __radd__(self, other):
        """Add a FermiSentence to a constant, i.e. `2 + FermiSentence({...})`"""

        if isinstance(other, (Number, ndarray)):
            return self.__add__(other)

        raise TypeError(f"Cannot add a FermiSentence to {type(other)}.")

    def __sub__(self, other):
        r"""Subtract a FermiSentence, FermiWord or constant from a FermiSentence"""
        if isinstance(other, FermiWord):
            other = FermiSentence({other: -1})
            return self.__add__(other)

        if isinstance(other, Number):
            other = FermiSentence({FermiWord({}): -1 * other})  # -constant * I
            return self.__add__(other)

        if isinstance(other, ndarray):
            if qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            other = FermiSentence({FermiWord({}): -1 * other})  # -constant * I
            return self.__add__(other)

        if isinstance(other, FermiSentence):
            other = FermiSentence(dict(zip(other.keys(), [-1 * v for v in other.values()])))
            return self.__add__(other)

        raise TypeError(f"Cannot subtract {type(other)} from a FermiSentence.")

    def __rsub__(self, other):
        """Subtract a FermiSentence to a constant, i.e.

        >>> 2 - FermiSentence({...})
        """

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            self_fs = FermiSentence(dict(zip(self.keys(), [-1 * v for v in self.values()])))
            other_fs = FermiSentence({FermiWord({}): other})  # constant * I
            return self_fs + other_fs

        raise TypeError(f"Cannot subtract a FermiSentence from {type(other)}.")

    def __mul__(self, other):
        r"""Multiply two Fermi sentences by iterating over each sentence and multiplying the Fermi
        words pair-wise"""

        if isinstance(other, FermiWord):
            other = FermiSentence({other: 1})

        if isinstance(other, FermiSentence):
            if (len(self) == 0) or (len(other) == 0):
                return FermiSentence({FermiWord({}): 0})

            product = FermiSentence({})

            for fw1, coeff1 in self.items():
                for fw2, coeff2 in other.items():
                    product[fw1 * fw2] += coeff1 * coeff2

            return product

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            vals = [i * other for i in self.values()]
            return FermiSentence(dict(zip(self.keys(), vals)))

        raise TypeError(f"Cannot multiply FermiSentence by {type(other)}.")

    def __rmul__(self, other):
        r"""Reverse multiply a FermiSentence

        Multiplies a FermiSentence "from the left" with an object that can't be modified
        to support __mul__ for FermiSentence. Will be defaulted in for example when
        multiplying ``2 * fermi_sentence``, since the ``__mul__`` operator on an integer
        will fail to multiply with a FermiSentence"""

        if isinstance(other, (Number, ndarray)):
            if isinstance(other, ndarray) and qml.math.size(other) > 1:
                raise ValueError(
                    f"Arithmetic Fermi operations can only accept an array of length 1, "
                    f"but received {other} of length {len(other)}"
                )
            vals = [i * other for i in self.values()]
            return FermiSentence(dict(zip(self.keys(), vals)))

        raise TypeError(f"Cannot multiply {type(other)} by FermiSentence.")

    def __pow__(self, value):
        r"""Exponentiate a Fermi sentence to an integer power."""
        if value < 0 or not isinstance(value, int):
            raise ValueError("The exponent must be a positive integer.")

        operator = FermiSentence({FermiWord({}): 1})  # 1 times Identity

        for _ in range(value):
            operator *= self

        return operator

    def simplify(self, tol=1e-8):
        r"""Remove any FermiWords in the FermiSentence with coefficients less than the threshold
        tolerance."""
        items = list(self.items())
        for fw, coeff in items:
            if abs(coeff) <= tol:
                del self[fw]