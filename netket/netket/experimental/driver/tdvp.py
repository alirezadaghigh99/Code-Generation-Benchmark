class TDVP(TDVPBaseDriver):
    """
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :class:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.

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
        qgt: LinearOperator = None,
        linear_solver=nk.optimizer.solver.pinv_smooth,
        linear_solver_restart: bool = False,
        error_norm: Union[str, Callable] = "euclidean",
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
            qgt: The QGT specification.
            linear_solver: The solver for solving the linear system determining the time evolution.
                This must be a jax-jittable function :code:`f(A,b) -> x` that accepts a Matrix-like, Linear Operator
                PyTree object :math:`A` and a vector-like PyTree :math:`b` and returns the PyTree :math:`x` solving
                the system :math:`Ax=b`.
                Defaults to :func:`nk.optimizer.solver.pinv_smooth` with the default svd threshold of 1e-10.
                To change the svd threshold you can use :func:`functools.partial` as follows:
                :code:`functools.partial(nk.optimizer.solver.pinv_smooth, rcond_smooth=1e-5)`.
            linear_solver_restart: If False (default), the last solution of the linear system
                is used as initial value in subsequent steps.
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
        """
        if qgt is None:
            qgt = QGTAuto(solver=linear_solver)

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

        self.qgt = qgt
        self.linear_solver = linear_solver
        self.linear_solver_restart = linear_solver_restart

        super().__init__(
            operator, variational_state, integrator, t0=t0, error_norm=error_norm
        )

