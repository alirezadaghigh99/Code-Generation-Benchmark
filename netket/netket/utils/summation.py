class KahanSum:
    """
    Accumulator implementing Kahan summation [1], which reduces
    the effect of accumulated floating-point error.

    [1] https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """

    value: Scalar
    compensator: Scalar = 0.0

    def __add__(self, other: Scalar):
        delta = other - self.compensator
        new_value = self.value + delta
        new_compensator = (new_value - self.value) - delta
        return KahanSum(new_value, new_compensator)

