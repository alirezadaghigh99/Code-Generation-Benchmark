class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """

    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return softplus(x)

    def _inverse(self, y):
        return softplus_inv(y)

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x)

