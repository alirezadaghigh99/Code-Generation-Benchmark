class CubicInterpolation(AbstractGlobalInterpolation):
    """Piecewise cubic spline interpolation over the interval $[t_0, t_1]$."""

    ts: Real[Array, " times"]
    # d, c, b, a
    coeffs: tuple[
        PyTree[Shaped[Array, "times-1 ?*shape"], "Y"],
        PyTree[Shaped[Array, "times-1 ?*shape"], "Y"],
        PyTree[Shaped[Array, "times-1 ?*shape"], "Y"],
        PyTree[Shaped[Array, "times-1 ?*shape"], "Y"],
    ]

    def __check_init__(self):
        def _check(d, c, b, a):
            error_msg = (
                "Each cubic coefficient must have `times - 1` entries, where "
                "`times = self.ts.shape[0]`."
            )
            if d.shape[0] + 1 != self.ts.shape[0]:
                raise ValueError(error_msg)
            if c.shape[0] + 1 != self.ts.shape[0]:
                raise ValueError(error_msg)
            if b.shape[0] + 1 != self.ts.shape[0]:
                raise ValueError(error_msg)
            if a.shape[0] + 1 != self.ts.shape[0]:
                raise ValueError(error_msg)

        jtu.tree_map(_check, *self.coeffs)

    @property
    def ts_size(self) -> IntScalarLike:
        return self.ts.shape[0]

    @eqx.filter_jit
    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> PyTree[Shaped[Array, "?*shape"], "Y"]:
        r"""Evaluate the cubic interpolation.

        **Arguments:**

        - `t0`: Any point in $[t_0, t_1]$ to evaluate the interpolation at.
        - `t1`: If passed, then the increment from `t1` to `t0` is evaluated instead.
        - `left`: Across jump points: whether to treat the path as left-continuous
            or right-continuous. [In practice cubic interpolation is always continuous
            except around `NaN`s.]

        !!! faq "FAQ"

            Note that we use $t_0$ and $t_1$ to refer to the overall interval, as
            obtained via `instance.t0` and `instance.t1`. We use `t0` and `t1` to refer
            to some subinterval of $[t_0, t_1]$. This is an API that is used for
            consistency with the rest of the package, and just happens to be a little
            confusing here.

        **Returns:**

        If `t1` is not passed:

        The interpolation of the data at `t0`.

        If `t1` is passed:

        The increment between `t0` and `t1`.
        """

        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        index, frac = self._interpret_t(t0, left)

        d, c, b, a = self.coeffs

        with jax.numpy_dtype_promotion("standard"):
            return (
                ω(a)[index]
                + frac * (ω(b)[index] + frac * (ω(c)[index] + frac * ω(d)[index]))
            ).ω

    @eqx.filter_jit
    def derivative(
        self, t: RealScalarLike, left: bool = True
    ) -> PyTree[Shaped[Array, "?*shape"], "Y"]:
        r"""Evaluate the derivative of the cubic interpolation. Essentially equivalent
        to `jax.jvp(self.evaluate, (t,), (jnp.ones_like(t),))`.


        **Arguments:**

        - `t`: Any point in $[t_0, t_1]$ to evaluate the derivative at.
        - `left`: Whether to obtain the left-derivative or right-derivative at that
            point. [In practice cubic interpolation is always continuously
            differentiable except around `NaN`s.]

        **Returns:**

        The derivative of the interpolation of the data.
        """

        index, frac = self._interpret_t(t, left)

        d, c, b, _ = self.coeffs

        with jax.numpy_dtype_promotion("standard"):
            return (ω(b)[index] + frac * (2 * ω(c)[index] + frac * 3 * ω(d)[index])).ω

