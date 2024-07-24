class SpectralNormError(AlgorithmicError):
    """Class representing the spectral norm error.

    The spectral norm error is defined as the distance, in spectral norm, between the true unitary
    we intend to apply and the approximate unitary that is actually applied.

    Args:
        error (float): The numerical value of the error

    **Example**

    >>> s1 = SpectralNormError(0.01)
    >>> s2 = SpectralNormError(0.02)
    >>> s1.combine(s2)
    SpectralNormError(0.03)
    """

    def __repr__(self):
        """Return formal string representation."""

        return f"SpectralNormError({self.error})"

    def combine(self, other: "SpectralNormError"):
        """Combine two spectral norm errors.

        Args:
            other (SpectralNormError): The other instance of error being combined.

        Returns:
            SpectralNormError: The total error after combination.

        **Example**

        >>> s1 = SpectralNormError(0.01)
        >>> s2 = SpectralNormError(0.02)
        >>> s1.combine(s2)
        SpectralNormError(0.03)
        """
        return self.__class__(self.error + other.error)

    @staticmethod
    def get_error(approximate_op: Operator, exact_op: Operator):
        """Compute spectral norm error between two operators.

        Args:
            approximate_op (Operator): The approximate operator.
            exact_op (Operator): The exact operator.

        Returns:
            float: The error between the exact operator and its
            approximation.

        **Example**

        >>> Op1 = qml.RY(0.40, 0)
        >>> Op2 = qml.RY(0.41, 0)
        >>> SpectralNormError.get_error(Op1, Op2)
        0.004999994791668309
        """
        wire_order = exact_op.wires
        m1 = qml.matrix(exact_op, wire_order=wire_order)
        m2 = qml.matrix(approximate_op, wire_order=wire_order)
        return qml.math.max(qml.math.svd(m1 - m2, compute_uv=False))

class SpectralNormError(AlgorithmicError):
    """Class representing the spectral norm error.

    The spectral norm error is defined as the distance, in spectral norm, between the true unitary
    we intend to apply and the approximate unitary that is actually applied.

    Args:
        error (float): The numerical value of the error

    **Example**

    >>> s1 = SpectralNormError(0.01)
    >>> s2 = SpectralNormError(0.02)
    >>> s1.combine(s2)
    SpectralNormError(0.03)
    """

    def __repr__(self):
        """Return formal string representation."""

        return f"SpectralNormError({self.error})"

    def combine(self, other: "SpectralNormError"):
        """Combine two spectral norm errors.

        Args:
            other (SpectralNormError): The other instance of error being combined.

        Returns:
            SpectralNormError: The total error after combination.

        **Example**

        >>> s1 = SpectralNormError(0.01)
        >>> s2 = SpectralNormError(0.02)
        >>> s1.combine(s2)
        SpectralNormError(0.03)
        """
        return self.__class__(self.error + other.error)

    @staticmethod
    def get_error(approximate_op: Operator, exact_op: Operator):
        """Compute spectral norm error between two operators.

        Args:
            approximate_op (Operator): The approximate operator.
            exact_op (Operator): The exact operator.

        Returns:
            float: The error between the exact operator and its
            approximation.

        **Example**

        >>> Op1 = qml.RY(0.40, 0)
        >>> Op2 = qml.RY(0.41, 0)
        >>> SpectralNormError.get_error(Op1, Op2)
        0.004999994791668309
        """
        wire_order = exact_op.wires
        m1 = qml.matrix(exact_op, wire_order=wire_order)
        m2 = qml.matrix(approximate_op, wire_order=wire_order)
        return qml.math.max(qml.math.svd(m1 - m2, compute_uv=False))

