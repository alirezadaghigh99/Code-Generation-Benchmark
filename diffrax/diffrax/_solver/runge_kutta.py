class MultiButcherTableau(eqx.Module):
    """Wraps multiple [`diffrax.ButcherTableau`][]s together. Used in some multi-tableau
    solvers, like IMEX methods.

    !!! important

        This API is not stable, and deliberately undocumented. (The reason is that we
        might yet adapt this to implement Stochastic Runge--Kutta methods.)
    """

    tableaus: tuple[ButcherTableau, ...]

    def __init__(self, *tableaus: ButcherTableau):
        self.tableaus = tableaus

