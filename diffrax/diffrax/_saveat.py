class SaveAt(eqx.Module):
    """Determines what to save as output from the differential equation solve.

    Instances of this class should be passed as the `saveat` argument of
    [`diffrax.diffeqsolve`][].
    """

    subs: PyTree[SubSaveAt] = None
    dense: bool = False
    solver_state: bool = False
    controller_state: bool = False
    made_jump: bool = False

    def __init__(
        self,
        *,
        t0: bool = False,
        t1: bool = False,
        ts: Union[None, Sequence[RealScalarLike], Real[Array, " times"]] = None,
        steps: bool = False,
        fn: Callable = save_y,
        subs: PyTree[SubSaveAt] = None,
        dense: bool = False,
        solver_state: bool = False,
        controller_state: bool = False,
        made_jump: bool = False,
    ):
        if subs is None:
            if t0 or t1 or (ts is not None) or steps:
                subs = SubSaveAt(t0=t0, t1=t1, ts=ts, steps=steps, fn=fn)
        else:
            if t0 or t1 or (ts is not None) or steps:
                raise ValueError(
                    "Cannot pass both `subs` and any of `t0`, `t1`, `ts`, `steps` to "
                    "`SaveAt`."
                )
        self.subs = subs
        self.dense = dense
        self.solver_state = solver_state
        self.controller_state = controller_state
        self.made_jump = made_jump

