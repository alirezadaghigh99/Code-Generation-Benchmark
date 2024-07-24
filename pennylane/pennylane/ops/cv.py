class TensorN(CVObservable):
    r"""
    The tensor product of the :class:`~.NumberOperator` acting on different wires.

    If a single wire is defined, returns a :class:`~.NumberOperator` instance for convenient gradient computations.

    When used with the :func:`~pennylane.expval` function, the expectation value
    :math:`\langle \hat{n}_{i_0} \hat{n}_{i_1}\dots \hat{n}_{i_{N-1}}\rangle`
    for a (sub)set of modes :math:`[i_0, i_1, \dots, i_{N-1}]` of the system is
    returned.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 0

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on

    .. details::
        :title: Usage Details

        Example for multiple modes:

        >>> cv_obs = qml.TensorN(wires=[0, 1])
        >>> cv_obs
        TensorN(wires=[0, 1])
        >>> cv_obs.ev_order is None
        True

        Example for a single mode (yields a :class:`~.NumberOperator`):

        >>> cv_obs = qml.TensorN(wires=[1])
        >>> cv_obs
        NumberOperator(wires=[1])
        >>> cv_obs.ev_order
        2
    """

    num_params = 0
    num_wires = AnyWires
    ev_order = None

    def __init__(self, wires):
        super().__init__(wires=wires)

    def __new__(cls, wires=None):
        # Custom definition for __new__ needed such that a NumberOperator can
        # be returned when a single mode is defined

        if wires is not None and (isinstance(wires, int) or len(wires) == 1):
            return NumberOperator(wires=wires)

        return super().__new__(cls)

    def label(self, decimals=None, base_label=None, cache=None):
        if base_label is not None:
            return base_label
        return "⊗".join("n" for _ in self.wires)

class NumberOperator(CVObservable):
    r"""
    The photon number observable :math:`\langle \hat{n}\rangle`.

    The number operator is defined as
    :math:`\hat{n} = \a^\dagger \a = \frac{1}{2\hbar}(\x^2 +\p^2) -\I/2`.

    When used with the :func:`~pennylane.expval` function, the mean
    photon number :math:`\braket{\hat{n}}` is returned.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Observable order: 2nd order in the quadrature operators
    * Heisenberg representation:

      .. math:: M = \frac{1}{2\hbar}\begin{bmatrix}
            -\hbar & 0 & 0\\
            0 & 1 & 0\\
            0 & 0 & 1
        \end{bmatrix}

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on
    """

    num_params = 0
    num_wires = 1

    ev_order = 2

    def __init__(self, wires):
        super().__init__(wires=wires)

    @staticmethod
    def _heisenberg_rep(p):
        hbar = 2
        return np.diag([-0.5, 0.5 / hbar, 0.5 / hbar])

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "n"

