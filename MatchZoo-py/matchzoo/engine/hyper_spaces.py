class quniform(HyperoptProxy):
    """:func:`hyperopt.hp.quniform` proxy."""

    def __init__(
        self,
        low: numbers.Number,
        high: numbers.Number,
        q: numbers.Number = 1
    ):
        """
        :func:`hyperopt.hp.quniform` proxy.

        If using with integer values, then `high` is exclusive.

        :param low: lower bound of the space
        :param high: upper bound of the space
        :param q: similar to the `step` in the python built-in `range`
        """
        super().__init__(hyperopt_func=hyperopt.hp.quniform,
                         low=low,
                         high=high, q=q)
        self._low = low
        self._high = high
        self._q = q

    def __str__(self):
        """:return: `str` representation of the hyper space."""
        return f'quantitative uniform distribution in  ' \
               f'[{self._low}, {self._high}), with a step size of {self._q}'

