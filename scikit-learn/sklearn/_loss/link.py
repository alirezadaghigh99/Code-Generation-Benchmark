def _inclusive_low_high(interval, dtype=np.float64):
    """Generate values low and high to be within the interval range.

    This is used in tests only.

    Returns
    -------
    low, high : tuple
        The returned values low and high lie within the interval.
    """
    eps = 10 * np.finfo(dtype).eps
    if interval.low == -np.inf:
        low = -1e10
    elif interval.low < 0:
        low = interval.low * (1 - eps) + eps
    else:
        low = interval.low * (1 + eps) + eps

    if interval.high == np.inf:
        high = 1e10
    elif interval.high < 0:
        high = interval.high * (1 + eps) - eps
    else:
        high = interval.high * (1 - eps) - eps

    return low, high

class Interval:
    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool

    def __post_init__(self):
        """Check that low <= high"""
        if self.low > self.high:
            raise ValueError(
                f"One must have low <= high; got low={self.low}, high={self.high}."
            )

    def includes(self, x):
        """Test whether all values of x are in interval range.

        Parameters
        ----------
        x : ndarray
            Array whose elements are tested to be in interval range.

        Returns
        -------
        result : bool
        """
        if self.low_inclusive:
            low = np.greater_equal(x, self.low)
        else:
            low = np.greater(x, self.low)

        if not np.all(low):
            return False

        if self.high_inclusive:
            high = np.less_equal(x, self.high)
        else:
            high = np.less(x, self.high)

        # Note: np.all returns numpy.bool_
        return bool(np.all(high))

