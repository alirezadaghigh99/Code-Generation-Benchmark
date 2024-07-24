class TDVPSchmitt(TDVPBaseDriver):
    r"""
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :class:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.

    This driver, which only works with standard MCState variational states, uses the regularization
    procedure described in `M. Schmitt's PRL 125 <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.125.100503>`_ .

    With the force vector

    .. math::

        F_k=\langle \mathcal O_{\theta_k}^* E_{loc}^{\theta}\rangle_c

    and the quantum Fisher matrix

    .. math::

        S_{k,k'} = \langle \mathcal O_{\theta_k} (\mathcal O_{\theta_{k'}})^*\rangle_c

    and for real parameters :math:`\theta\in\mathbb R`, the TDVP equation reads

    .. math::

        q\big[S_{k,k'}\big]\theta_{k'} = -q\big[xF_k\big]

    Here, either :math:`q=\text{Re}` or :math:`q=\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.

    For ground state search a regularization controlled by a parameter :math:`\rho` can be included
    by increasing the diagonal entries and solving

    .. math::

        q\big[(1+\rho\delta_{k,k'})S_{k,k'}\big]\theta_{k'} = -q\big[F_k\big]

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

    .. math::

        S = V\Sigma V^\dagger

    with a diagonal matrix :math:`\Sigma_{kk}=\sigma_k`
    Assuming that :math:`\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed
    from the regularized inverted eigenvalues

    .. math::

        \tilde\sigma_k^{-1}=\frac{1}{\Big(1+\big(\frac{\epsilon_{SVD}}{\sigma_j/\sigma_1}\big)^6\Big)\Big(1+\big(\frac{\epsilon_{SNR}}{\text{SNR}(\rho_k)}\big)^6\Big)}

    with :math:`\text{SNR}(\rho_k)` the signal-to-noise ratio of
    :math:`\rho_k=V_{k,k'}^{\dagger}F_{k'}` (see
    `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).


    .. note::

        This TDVP Driver uses the time-integrators from the `nkx.dynamics` module, which are
        automatically executed under a `jax.jit` context.

        When running computations on GPU, this can lead to infinite hangs or extremely long
        compilation times. In those cases, you might try setting the configuration variable
        `nk.config.netket_experimental_disable_ode_jit = True` to mitigate those issues.

    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        integrator: RKIntegratorConfig,
        *,
        t0: float = 0.0,
        propagation_type: str = "real",
        holomorphic: Optional[bool] = None,
        diag_shift: float = 0.0,
        diag_scale: Optional[float] = None,
        error_norm: Union[str, Callable] = "qgt",
        rcond: float = 1e-14,
        rcond_smooth: float = 1e-8,
        snr_atol: float = 1,
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (Hamiltonian for pure states,
                Lindbladian for density operators).
            variational_state: The variational state.
            integrator: Configuration of the algorithm used for solving the ODE.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schrödinger equation (SE), "imag" for the imaginary-time SE.
            error_norm: Norm function used to calculate the error with adaptive integrators.
                Can be either "euclidean" for the standard L2 vector norm :math:`w^\dagger w`,
                "maximum" for the maximum norm :math:`\max_i |w_i|`
                or "qgt", in which case the scalar product induced by the QGT :math:`S` is used
                to compute the norm :math:`\Vert w \Vert^2_S = w^\dagger S w` as suggested
                in PRL 125, 100503 (2020).
                Additionally, it possible to pass a custom function with signature
                :code:`norm(x: PyTree) -> float`
                which maps a PyTree of parameters :code:`x` to the corresponding norm.
                Note that norm is used in jax.jit-compiled code.
            holomorphic: a flag to indicate that the wavefunction is holomorphic.
            diag_shift: diagonal shift of the quantum geometric tensor (QGT)
            diag_scale: If not None rescales the diagonal shift of the QGT
            rcond : Cut-off ratio for small singular :math:`\sigma_k` values of the
                Quantum Geometric Tensor. For the purposes of rank determination,
                singular values are treated as zero if they are smaller than rcond times
                the largest singular value :code:`\sigma_{max}`.
            rcond_smooth : Smooth cut-off ratio for singular values of the Quantum Geometric
                Tensor. This regularization parameter used with a similar effect to `rcond`
                but with a softer curve. See :math:`\epsilon_{SVD}` in the formula
                above.
            snr_atol: Noise regularisation absolute tolerance, meaning that eigenvalues of
                the S matrix that have a signal to noise ratio above this quantity will be
                (soft) truncated. This is :math:`\epsilon_{SNR}` in the formulas above.

        """
        self.propagation_type = propagation_type
        if isinstance(variational_state, VariationalMixedState):
            # assuming Lindblad Dynamics
            # TODO: support density-matrix imaginary time evolution
            if propagation_type == "real":
                self._loss_grad_factor = 1.0
            else:
                raise ValueError(
                    "only real-time Lindblad evolution is supported for " "mixed states"
                )
        else:
            if propagation_type == "real":
                self._loss_grad_factor = -1.0j
            elif propagation_type == "imag":
                self._loss_grad_factor = -1.0
            else:
                raise ValueError("propagation_type must be one of 'real', 'imag'")

        self.rcond = rcond
        self.rcond_smooth = rcond_smooth
        self.snr_atol = snr_atol

        self.diag_shift = diag_shift
        self.holomorphic = holomorphic
        self.diag_scale = diag_scale

        super().__init__(
            operator, variational_state, integrator, t0=t0, error_norm=error_norm
        )

