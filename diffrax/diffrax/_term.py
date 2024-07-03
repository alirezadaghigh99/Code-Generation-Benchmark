class ODETerm(AbstractTerm[_VF, RealScalarLike]):
    r"""A term representing $f(t, y(t), args) \mathrm{d}t$. That is to say, the term
    appearing on the right hand side of an ODE, in which the control is time.

    `vector_field` should return some PyTree, with the same structure as the initial
    state `y0`, and with every leaf shape-broadcastable and dtype-upcastable to the
    equivalent leaf in `y0`.

    !!! example

        ```python
        vector_field = lambda t, y, args: -y
        ode_term = ODETerm(vector_field)
        diffeqsolve(ode_term, ...)
        ```
    """

    vector_field: Callable[[RealScalarLike, Y, Args], _VF]

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> _VF:
        out = self.vector_field(t, y, args)
        if jtu.tree_structure(out) != jtu.tree_structure(y):
            raise ValueError(
                "The vector field inside `ODETerm` must return a pytree with the "
                "same structure as `y0`."
            )

        def _broadcast_and_upcast(oi, yi):
            oi = jnp.broadcast_to(oi, jnp.shape(yi))
            oi = upcast_or_raise(
                oi,
                yi,
                "the vector field passed to `ODETerm`",
                "the corresponding leaf of `y`",
            )
            return oi

        return jtu.tree_map(_broadcast_and_upcast, out, y)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        return t1 - t0

    def prod(self, vf: _VF, control: RealScalarLike) -> Y:
        def _mul(v):
            c = upcast_or_raise(
                control,
                v,
                "the output of `ODETerm.contr(...)`",
                "the output of `ODETerm.vf(...)`",
            )
            return c * v

        return jtu.tree_map(_mul, vf)