class BooleanFn:
    r"""Wrapper for simple callables with Boolean output that can be
    manipulated and combined with bit-wise operators.

    Args:
        fn (callable): Function to be wrapped. It can accept any number
            of arguments, and must return a Boolean.

    **Example**

    Consider functions that filter numbers to lie in a certain domain.
    We may wrap them using ``BooleanFn``:

    >>> bigger_than_4 = qml.BooleanFn(lambda x: x > 4)
    >>> smaller_than_10 = qml.BooleanFn(lambda x: x < 10)
    >>> is_int = qml.BooleanFn(lambda x: isinstance(x, int))
    >>> bigger_than_4(5.2)
    True
    >>> smaller_than_10(20.1)
    False
    >>> is_int(2.3)
    False

    These can then be combined into a single callable using boolean operators,
    such as ``&``, logical and:

    >>> between_4_and_10 = bigger_than_4 & smaller_than_10
    >>> between_4_and_10(-3.2)
    False
    >>> between_4_and_10(9.9)
    True
    >>> between_4_and_10(19.7)
    False

    Other supported operators are ``|``, logical or, and ``~``, logical not:

    >>> smaller_equal_than_4 = ~bigger_than_4
    >>> smaller_than_10_or_int = smaller_than_10 | is_int

    .. warning::

        Note that Python conditional expressions are evaluated from left to right.
        As a result, the order of composition may matter, even though logical
        operators such as ``|`` and ``&`` are symmetric.

        For example:

        >>> is_int = qml.BooleanFn(lambda x: isinstance(x, int))
        >>> has_bit_length_3 = qml.BooleanFn(lambda x: x.bit_length()==3)
        >>> (is_int & has_bit_length_3)(4)
        True
        >>> (is_int & has_bit_length_3)(2.3)
        False
        >>> (has_bit_length_3 & is_int)(2.3)
        AttributeError: 'float' object has no attribute 'bit_length'

    """

    def __init__(self, fn, name=None):
        self.fn = fn
        self.name = name or self.fn.__name__
        functools.update_wrapper(self, fn)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __xor__(self, other):
        return Xor(self, other)

    def __invert__(self):
        return Not(self)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"BooleanFn({self.name})" if not (self.bitwise or self.conditional) else self.name

    @property
    def bitwise(self):
        """Determine whether wrapped callable performs a bit-wise operation or not.
        This checks for the ``operands`` attribute that should be defined by it."""
        return bool(getattr(self, "operands", tuple()))

    @property
    def conditional(self):
        """Determine whether wrapped callable is for a conditional or not.
        This checks for the ``condition`` attribute that should be defined by it."""
        return bool(getattr(self, "condition", None))