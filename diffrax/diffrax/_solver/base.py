class HalfSolver(
    AbstractAdaptiveSolver[_SolverState], AbstractWrappedSolver[_SolverState]
):
    """Wraps another solver, trading cost in order to provide error estimates. (That
    is, it means the solver can be used with an adaptive step size controller,
    regardless of whether the underlying solver supports adaptive step sizing.)

    For every step of the wrapped solver, it does this by also making two half-steps,
    and comparing the results between the full step and the two half steps. Hence the
    name "HalfSolver".

    As such each step costs 3 times the computational cost of the wrapped solver,
    whilst producing results that are roughly twice as accurate, in addition to
    producing error estimates.

    !!! tip

        Many solvers already provide error estimates, making `HalfSolver` primarily
        useful when using a solver that doesn't provide error estimates -- e.g.
        [`diffrax.Euler`][]. Such solvers are most common when solving SDEs.
    """

    solver: AbstractSolver[_SolverState]

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def error_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        if is_sde(terms):
            order = self.strong_order(terms)
            if order is not None:
                order = order + 0.5
        else:
            order = self.order(terms)
            if order is not None:
                order = order + 1
        return order

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return self.solver.init(terms, t0, t1, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        original_solver_state = solver_state
        thalf = t0 + 0.5 * (t1 - t0)

        yhalf, _, _, solver_state, result1 = self.solver.step(
            terms, t0, thalf, y0, args, solver_state, made_jump
        )
        y1, _, _, solver_state, result2 = self.solver.step(
            terms, thalf, t1, yhalf, args, solver_state, made_jump=False
        )

        # TODO: use dense_info from the pair of half-steps instead
        y1_alt, _, dense_info, _, result3 = self.solver.step(
            terms, t0, t1, y0, args, original_solver_state, made_jump
        )

        y_error = (y1**ω - y1_alt**ω).call(jnp.abs).ω
        result = update_result(result1, update_result(result2, result3))

        return y1, y_error, dense_info, solver_state, result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)

