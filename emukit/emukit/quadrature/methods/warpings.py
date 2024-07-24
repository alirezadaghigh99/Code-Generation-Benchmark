class SquareRootWarping(Warping):
    r"""The square root warping.

    .. math::
        w(y)=\begin{cases}
                  c + \frac{1}{2}y^2 & \text{is_inverted is False (default)}\\
                  c - \frac{1}{2}y^2 &\text{otherwise}
              \end{cases},

    where :math:`c` is a constant.

    :param offset: The offset :math:`c` of the warping.
    :param is_inverted: Inverts the warping if ``True``. Default is ``False``.

    """

    def __init__(self, offset: float, is_inverted: Optional[bool] = False):
        self.offset = offset
        self.is_inverted = is_inverted

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.is_inverted:
            return self.offset - 0.5 * (Y * Y)
        else:
            return self.offset + 0.5 * (Y * Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        if self.is_inverted:
            return np.sqrt(2.0 * (self.offset - Y))
        else:
            return np.sqrt(2.0 * (Y - self.offset))

class IdentityWarping(Warping):
    """The identity warping

    .. math::
        w(y) = y.

    """

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return Y

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        return Y

