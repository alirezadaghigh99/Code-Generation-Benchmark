def diffeqsolve(
    terms: PyTree[AbstractTerm],
    solver: AbstractSolver,
    t0: RealScalarLike,
    t1: RealScalarLike,
    dt0: Optional[RealScalarLike],
    y0: PyTree[ArrayLike],
    args: PyTree[Any] = None,
    *,
    saveat: SaveAt = SaveAt(t1=True),
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
    event: Optional[Event] = None,
    max_steps: Optional[int] = 4096,
    throw: bool = True,
    progress_meter: AbstractProgressMeter = NoProgressMeter(),
    solver_state: Optional[PyTree[ArrayLike]] = None,
    controller_state: Optional[PyTree[ArrayLike]] = None,
    made_jump: Optional[BoolScalarLike] = None,
    # Exists for backward compatibility
    discrete_terminating_event: Optional[AbstractDiscreteTerminatingEvent] = None,
) -> Solution:
    """Solves a differential equation.

    This function is the main entry point for solving all kinds of initial value
    problems, whether they are ODEs, SDEs, or CDEs.

    The differential equation is integrated from `t0` to `t1`.

    See the [Getting started](../usage/getting-started.md) page for example usage.

    **Main arguments:**

    These are the arguments most commonly used day-to-day.

    - `terms`: The terms of the differential equation. This specifies the vector field.
        (For non-ordinary differential equations (SDEs, CDEs), this also specifies the
        Brownian motion or the control.)
    - `solver`: The solver for the differential equation. See the guide on [how to
        choose a solver](../usage/how-to-choose-a-solver.md).
    - `t0`: The start of the region of integration.
    - `t1`: The end of the region of integration.
    - `dt0`: The step size to use for the first step. If using fixed step sizes then
        this will also be the step size for all other steps. (Except the last one,
        which may be slightly smaller and clipped to `t1`.) If set as `None` then the
        initial step size will be determined automatically.
    - `y0`: The initial value. This can be any PyTree of JAX arrays. (Or types that
        can be coerced to JAX arrays, like Python floats.)
    - `args`: Any additional arguments to pass to the vector field.
    - `saveat`: What times to save the solution of the differential equation. See
        [`diffrax.SaveAt`][]. Defaults to just the last time `t1`. (Keyword-only
        argument.)
    - `stepsize_controller`: How to change the step size as the integration progresses.
        See the [list of stepsize controllers](../api/stepsize_controller.md).
        Defaults to using a fixed constant step size. (Keyword-only argument.)

    **Other arguments:**

    These arguments are less frequently used, and for most purposes you shouldn't need
    to understand these. All of these are keyword-only arguments.

    - `adjoint`: How to differentiate `diffeqsolve`. Defaults to
        discretise-then-optimise, which is usually the best option for most problems.
        See the page on [Adjoints](./adjoints.md) for more information.

    - `event`: An event at which to terminate the solve early. See the page on
        [Events](./events.md) for more information.

    - `max_steps`: The maximum number of steps to take before quitting the computation
        unconditionally.

        Can also be set to `None` to allow an arbitrary number of steps, although this
        is incompatible with `saveat=SaveAt(steps=True)` or `saveat=SaveAt(dense=True)`.

    - `throw`: Whether to raise an exception if the integration fails for any reason.

        If `True` then an integration failure will raise a runtime error.

        If `False` then the returned solution object will have a `result` field
        indicating whether any failures occurred.

        Possible failures include for example hitting `max_steps`, or the problem
        becoming too stiff to integrate. (For most purposes these failures are
        unusual.)

        !!! note

            When `jax.vmap`-ing a differential equation solve, then
            `throw=True` means that an exception will be raised if any batch element
            fails. You may prefer to set `throw=False` and inspect the `result` field
            of the returned solution object, to determine which batch elements
            succeeded and which failed.

    - `progress_meter`: A progress meter to indicate how far through the solve has
        progressed. See [the progress meters page](./progress_meter.md).

    - `solver_state`: Some initial state for the solver. Generally obtained by
        `SaveAt(solver_state=True)` from a previous solve.

    - `controller_state`: Some initial state for the step size controller. Generally
        obtained by `SaveAt(controller_state=True)` from a previous solve.

    - `made_jump`: Whether a jump has just been made at `t0`. Used to update
        `solver_state` (if passed). Generally obtained by `SaveAt(made_jump=True)`
        from a previous solve.

    **Returns:**

    Returns a [`diffrax.Solution`][] object specifying the solution to the differential
    equation.

    **Raises:**

    - `ValueError` for bad inputs.
    - `RuntimeError` if `throw=True` and the integration fails (e.g. hitting the
        maximum number of steps).

    !!! note

        It is possible to have `t1 < t0`, in which case integration proceeds backwards
        in time.
    """

    #
    # Initial set-up
    #

    # Backward compatibility
    if discrete_terminating_event is not None:
        warnings.warn(
            "`diffrax.diffeqsolve(..., discrete_terminating_event=...)` is deprecated "
            "in favour of the more general `diffrax.diffeqsolve(..., event=...)` "
            "interface. This will be removed in some future version of Diffrax.",
            stacklevel=2,
        )
        if event is None:
            event = Event(
                cond_fn=DiscreteTerminatingEventToCondFn(discrete_terminating_event)
            )
        else:
            raise ValueError(
                "Cannot pass both "
                "`diffrax.diffeqsolve(..., event=..., discrete_terminating_event=...)`."
            )

    # Error checking
    if dt0 is not None:
        msg = (
            "Must have (t1 - t0) * dt0 >= 0, we instead got "
            f"t1 with value {t1} and type {type(t1)}, "
            f"t0 with value {t0} and type {type(t0)}, "
            f"dt0 with value {dt0} and type {type(dt0)}"
        )
        with jax.ensure_compile_time_eval(), jax.numpy_dtype_promotion("standard"):
            pred = (t1 - t0) * dt0 < 0
        dt0 = eqxi.error_if(jnp.array(dt0), pred, msg)

    # Error checking and warning for complex dtypes
    if any(
        eqx.is_array(xi) and jnp.iscomplexobj(xi)
        for xi in jtu.tree_leaves((terms, y0, args))
    ):
        warnings.warn(
            "Complex dtype support is work in progress, please read "
            "https://github.com/patrick-kidger/diffrax/pull/197 and proceed carefully.",
            stacklevel=2,
        )

    # Allow setting e.g. t0 as an int with dt0 as a float.
    timelikes = [t0, t1, dt0] + [
        s.ts for s in jtu.tree_leaves(saveat.subs, is_leaf=_is_subsaveat)
    ]
    timelikes = [x for x in timelikes if x is not None]
    with jax.numpy_dtype_promotion("standard"):
        time_dtype = jnp.result_type(*timelikes)
    if jnp.issubdtype(time_dtype, jnp.complexfloating):
        raise ValueError(
            "Cannot use complex dtype for `t0`, `t1`, `dt0`, or `SaveAt(ts=...)`."
        )
    elif jnp.issubdtype(time_dtype, jnp.floating):
        pass
    elif jnp.issubdtype(time_dtype, jnp.integer):
        time_dtype = lxi.default_floating_dtype()
    else:
        raise ValueError(f"Unrecognised time dtype {time_dtype}.")
    t0 = jnp.asarray(t0, dtype=time_dtype)
    t1 = jnp.asarray(t1, dtype=time_dtype)
    if dt0 is not None:
        dt0 = jnp.asarray(dt0, dtype=time_dtype)

    def _get_subsaveat_ts(saveat):
        out = [s.ts for s in jtu.tree_leaves(saveat.subs, is_leaf=_is_subsaveat)]
        return [x for x in out if x is not None]

    saveat = eqx.tree_at(
        _get_subsaveat_ts,
        saveat,
        replace_fn=lambda ts: ts.astype(time_dtype),  # noqa: F821
    )

    # Time will affect state, so need to promote the state dtype as well if necessary.
    # fixing issue with float64 and weak dtypes, see discussion at:
    # https://github.com/patrick-kidger/diffrax/pull/197#discussion_r1130173527
    def _promote(yi):
        with jax.numpy_dtype_promotion("standard"):
            _dtype = jnp.result_type(yi, time_dtype)  # noqa: F821
        return jnp.asarray(yi, dtype=_dtype)

    y0 = jtu.tree_map(_promote, y0)
    del timelikes

    # Backward compatibility
    if isinstance(
        solver, (EulerHeun, ItoMilstein, StratonovichMilstein)
    ) and _term_compatible(
        y0, args, terms, (ODETerm, AbstractTerm), solver.term_compatible_contr_kwargs
    ):
        warnings.warn(
            "Passing `terms=(ODETerm(...), SomeOtherTerm(...))` to "
            f"{solver.__class__.__name__} is deprecated in favour of "
            "`terms=MultiTerm(ODETerm(...), SomeOtherTerm(...))`. This means that "
            "the same terms can now be passed used for both general and SDE-specific "
            "solvers!",
            stacklevel=2,
        )
        terms = MultiTerm(*terms)

    # Error checking
    if not _term_compatible(
        y0, args, terms, solver.term_structure, solver.term_compatible_contr_kwargs
    ):
        raise ValueError(
            "`terms` must be a PyTree of `AbstractTerms` (such as `ODETerm`), with "
            f"structure {solver.term_structure}"
        )

    if is_sde(terms):
        if not isinstance(solver, (AbstractItoSolver, AbstractStratonovichSolver)):
            warnings.warn(
                f"`{type(solver).__name__}` is not marked as converging to either the "
                "Itô or the Stratonovich solution.",
                stacklevel=2,
            )
        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            # Specific check to not work even if using HalfSolver(Euler())
            if isinstance(solver, Euler):
                raise ValueError(
                    "An SDE should not be solved with adaptive step sizes with Euler's "
                    "method, as it may not converge to the correct solution."
                )
    if is_unsafe_sde(terms):
        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            raise ValueError(
                "`UnsafeBrownianPath` cannot be used with adaptive step sizes."
            )

    # Normalises time: if t0 > t1 then flip things around.
    direction = jnp.where(t0 < t1, 1, -1)
    t0 = t0 * direction
    t1 = t1 * direction
    if dt0 is not None:
        dt0 = dt0 * direction
    saveat = eqx.tree_at(
        _get_subsaveat_ts, saveat, replace_fn=lambda ts: ts * direction
    )
    stepsize_controller = stepsize_controller.wrap(direction)

    def _wrap(term):
        assert isinstance(term, AbstractTerm)
        assert not isinstance(term, MultiTerm)
        return WrapTerm(term, direction)

    terms = jtu.tree_map(
        _wrap,
        terms,
        is_leaf=lambda x: isinstance(x, AbstractTerm) and not isinstance(x, MultiTerm),
    )

    if isinstance(solver, AbstractImplicitSolver):

        def _get_tols(x):
            outs = []
            for attr in ("rtol", "atol", "norm"):
                if getattr(solver.root_finder, attr) is use_stepsize_tol:  # pyright: ignore
                    outs.append(getattr(x, attr))
            return tuple(outs)

        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            solver = eqx.tree_at(
                lambda s: _get_tols(s.root_finder),
                solver,
                _get_tols(stepsize_controller),
            )
        else:
            if len(_get_tols(solver.root_finder)) > 0:
                raise ValueError(
                    "A fixed step size controller is being used alongside an implicit "
                    "solver, but the tolerances for the implicit solver have not been "
                    "specified. (Being unspecified is the default in Diffrax.)\n"
                    "The correct fix is almost always to use an adaptive step size "
                    "controller. For example "
                    "`diffrax.diffeqsolve(..., "
                    "stepsize_controller=diffrax.PIDController(rtol=..., atol=...))`. "
                    "In this case the same tolerances are used for the implicit "
                    "solver as are used to control the adaptive stepping.\n"
                    "(Note for advanced users: the tolerances for the implicit "
                    "solver can also be explicitly set instead. For example "
                    "`diffrax.diffeqsolve(..., solver=diffrax.Kvaerno5(root_finder="
                    "diffrax.VeryChord(rtol=..., atol=..., "
                    "norm=optimistix.max_norm)))`. In this case the norm must also be "
                    "explicitly specified.)\n"
                    "Adaptive step size controllers are the preferred solution, as "
                    "sometimes the implicit solver may fail to converge, and in this "
                    "case an adaptive step size controller can reject the step and try "
                    "a smaller one, whilst with a fixed step size controller the "
                    "overall differential equation solve will simply fail."
                )

    # Error checking
    def _check_subsaveat_ts(ts):
        ts = eqxi.error_if(
            ts,
            ts[1:] < ts[:-1],
            "saveat.ts must be increasing or decreasing.",
        )
        ts = eqxi.error_if(
            ts,
            (ts > t1) | (ts < t0),
            "saveat.ts must lie between t0 and t1.",
        )
        return ts

    saveat = eqx.tree_at(_get_subsaveat_ts, saveat, replace_fn=_check_subsaveat_ts)

    def _subsaveat_direction_fn(x):
        if _is_subsaveat(x):
            if x.fn is not save_y:
                direction_fn = lambda t, y, args: x.fn(direction * t, y, args)
                return eqx.tree_at(lambda x: x.fn, x, direction_fn)
            else:
                return x
        else:
            return x

    saveat = jtu.tree_map(_subsaveat_direction_fn, saveat, is_leaf=_is_subsaveat)

    # Initialise states
    tprev = t0
    error_order = solver.error_order(terms)
    if controller_state is None:
        passed_controller_state = False
        (tnext, controller_state) = stepsize_controller.init(
            terms, t0, t1, y0, dt0, args, solver.func, error_order
        )
    else:
        passed_controller_state = True
        if dt0 is None:
            (tnext, _) = stepsize_controller.init(
                terms, t0, t1, y0, dt0, args, solver.func, error_order
            )
        else:
            tnext = t0 + dt0
    tnext = jnp.minimum(tnext, t1)
    if solver_state is None:
        passed_solver_state = False
        solver_state = solver.init(terms, t0, tnext, y0, args)
    else:
        passed_solver_state = True

    # Allocate memory to store output.
    def _allocate_output(subsaveat: SubSaveAt) -> SaveState:
        out_size = 0
        if subsaveat.t0:
            out_size += 1
        if subsaveat.ts is not None:
            out_size += len(subsaveat.ts)
        if subsaveat.steps:
            # We have no way of knowing how many steps we'll actually end up taking, and
            # XLA doesn't support dynamic shapes. So we just have to allocate the
            # maximum amount of steps we can possibly take.
            if max_steps is None:
                raise ValueError(
                    "`max_steps=None` is incompatible with saving at `steps=True`"
                )
            out_size += max_steps
        if subsaveat.t1 and not subsaveat.steps:
            out_size += 1
        saveat_ts_index = 0
        save_index = 0
        ts = jnp.full(out_size, direction * jnp.inf, dtype=time_dtype)
        struct = eqx.filter_eval_shape(subsaveat.fn, t0, y0, args)
        ys = jtu.tree_map(
            lambda y: jnp.full((out_size,) + y.shape, jnp.inf, dtype=y.dtype), struct
        )
        return SaveState(
            ts=ts, ys=ys, save_index=save_index, saveat_ts_index=saveat_ts_index
        )

    save_state = jtu.tree_map(_allocate_output, saveat.subs, is_leaf=_is_subsaveat)
    num_steps = 0
    num_accepted_steps = 0
    num_rejected_steps = 0
    made_jump = False if made_jump is None else made_jump
    result = RESULTS.successful
    if saveat.dense or event is not None:
        _, _, dense_info_struct, _, _ = eqx.filter_eval_shape(
            solver.step, terms, tprev, tnext, y0, args, solver_state, made_jump
        )
    if saveat.dense:
        if max_steps is None:
            raise ValueError(
                "`max_steps=None` is incompatible with `saveat.dense=True`"
            )
        dense_ts = jnp.full(max_steps + 1, jnp.inf, dtype=time_dtype)
        _make_full = lambda x: jnp.full(
            (max_steps,) + jnp.shape(x), jnp.inf, dtype=x.dtype
        )
        dense_infos = jtu.tree_map(_make_full, dense_info_struct)  # pyright: ignore[reportPossiblyUnboundVariable]
        dense_save_index = 0
    else:
        dense_ts = None
        dense_infos = None
        dense_save_index = None

    # Progress meter
    progress_meter_state = progress_meter.init()

    # Events
    if event is None:
        event_tprev = None
        event_tnext = None
        event_dense_info = None
        event_values = None
        event_mask = None
    else:
        event_tprev = tprev
        event_tnext = tnext
        # Fill the dense-info with dummy values on the first step, when we haven't yet
        # made any steps.
        # Note that we're threading a needle here! What if we terminate on the very
        # first step? Our dense-info (and thus a subsequent root find) will be
        # completely wrong!
        # Fortunately, this can't quite happen:
        # - A boolean event never uses dense-info (the interpolation is unused and we go
        #   to the end of the interval).
        # - A floating event can't terminate on the first step (it requires a sign
        #   change).
        event_dense_info = jtu.tree_map(
            lambda x: jnp.empty(x.shape, x.dtype),
            dense_info_struct,  # pyright: ignore[reportPossiblyUnboundVariable]
        )

        def _outer_cond_fn(cond_fn_i):
            event_value_i = cond_fn_i(
                tprev,
                y0,
                args,
                terms=terms,
                solver=solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
            )
            if jtu.tree_structure(event_value_i) != jtu.tree_structure(0):
                raise ValueError(
                    "Event functions must return a scalar, got PyTree with shape "
                    f"{jtu.tree_structure(event_value_i)}."
                )
            if jnp.shape(event_value_i) != ():
                raise ValueError(
                    "Event functions must return a scalar, got shape "
                    f"{jnp.shape(event_value_i)}."
                )
            event_dtype = jnp.result_type(event_value_i)
            if jnp.issubdtype(event_dtype, jnp.floating):
                event_mask_i = False  # Has not yet had the opportunity to change sign.
            elif jnp.issubdtype(event_dtype, jnp.bool_):
                event_mask_i = event_value_i
            else:
                raise ValueError(
                    "Event functions must return either a boolean or a float, got "
                    f"{event_dtype}."
                )
            return event_value_i, event_mask_i

        event_values__mask = jtu.tree_map(
            _outer_cond_fn,
            event.cond_fn,
            is_leaf=callable,
        )
        event_structure = jtu.tree_structure(event.cond_fn, is_leaf=callable)
        event_values, event_mask = jtu.tree_transpose(
            event_structure,
            jtu.tree_structure((0, 0)),
            event_values__mask,
        )
        had_event = False
        event_mask_leaves = []
        for event_mask_i in jtu.tree_leaves(event_mask):
            event_mask_leaves.append(event_mask_i & jnp.invert(had_event))
            had_event = event_mask_i | had_event
        event_mask = jtu.tree_unflatten(event_structure, event_mask_leaves)
        result = RESULTS.where(
            had_event,
            RESULTS.event_occurred,
            result,
        )
        del had_event, event_structure, event_mask_leaves, event_values__mask

    # Initialise state
    init_state = State(
        y=y0,
        tprev=tprev,
        tnext=tnext,
        made_jump=made_jump,
        solver_state=solver_state,
        controller_state=controller_state,
        result=result,
        num_steps=num_steps,
        num_accepted_steps=num_accepted_steps,
        num_rejected_steps=num_rejected_steps,
        save_state=save_state,
        dense_ts=dense_ts,
        dense_infos=dense_infos,
        dense_save_index=dense_save_index,
        progress_meter_state=progress_meter_state,
        event_tprev=event_tprev,
        event_tnext=event_tnext,
        event_dense_info=event_dense_info,
        event_values=event_values,
        event_mask=event_mask,
    )

    #
    # Main loop
    #

    final_state, aux_stats = adjoint.loop(
        args=args,
        terms=terms,
        solver=solver,
        stepsize_controller=stepsize_controller,
        event=event,
        saveat=saveat,
        t0=t0,
        t1=t1,
        dt0=dt0,
        max_steps=max_steps,
        init_state=init_state,
        throw=throw,
        passed_solver_state=passed_solver_state,
        passed_controller_state=passed_controller_state,
        progress_meter=progress_meter,
    )

    #
    # Finish up
    #

    progress_meter.close(final_state.progress_meter_state)
    is_save_state = lambda x: isinstance(x, SaveState)
    ts = jtu.tree_map(
        lambda s: s.ts * direction, final_state.save_state, is_leaf=is_save_state
    )
    ys = jtu.tree_map(lambda s: s.ys, final_state.save_state, is_leaf=is_save_state)

    # It's important that we don't do any further postprocessing on `ys` here, as
    # it is the `final_state` value that is used when backpropagating via
    # `BacksolveAdjoint`.

    if saveat.controller_state:
        controller_state = final_state.controller_state
    else:
        controller_state = None
    if saveat.solver_state:
        solver_state = final_state.solver_state
    else:
        solver_state = None
    if saveat.made_jump:
        made_jump = final_state.made_jump
    else:
        made_jump = None
    if saveat.dense:
        interpolation = DenseInterpolation(
            ts=final_state.dense_ts,
            ts_size=final_state.dense_save_index + 1,
            infos=final_state.dense_infos,
            interpolation_cls=solver.interpolation_cls,
            direction=direction,
            t0_if_trivial=t0,
            y0_if_trivial=y0,
        )
    else:
        interpolation = None

    t0 = t0 * direction
    t1 = t1 * direction

    # Store metadata
    stats = {
        "num_steps": final_state.num_steps,
        "num_accepted_steps": final_state.num_accepted_steps,
        "num_rejected_steps": final_state.num_rejected_steps,
        "max_steps": max_steps,
        **aux_stats,
    }
    result = final_state.result
    event_mask = final_state.event_mask
    sol = Solution(
        t0=t0,
        t1=t1,
        ts=ts,
        ys=ys,
        interpolation=interpolation,
        stats=stats,
        result=result,
        solver_state=solver_state,
        controller_state=controller_state,
        made_jump=made_jump,
        event_mask=event_mask,
    )

    if throw:
        sol = result.error_if(sol, jnp.invert(is_okay(result)))
    return sol