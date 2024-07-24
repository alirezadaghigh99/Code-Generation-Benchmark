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

class WeaklyDiagonalControlTerm(_AbstractControlTerm[_VF, _Control]):
    r"""
    DEPRECATED. Prefer:

    ```python
    def vector_field(t, y, args):
        return lineax.DiagonalLinearOperator(...)

    diffrax.ControlTerm(vector_field, ...)
    ```

    ---

    A term representing the case of $f(t, y(t), args) \mathrm{d}x(t)$, in
    which the vector field - control interaction is a matrix-vector product, and the
    matrix is square and diagonal. In this case we may represent the matrix as a vector
    of just its diagonal elements. The matrix-vector product may be calculated by
    pointwise multiplying this vector with the control; this is more computationally
    efficient than writing out the full matrix and then doing a full matrix-vector
    product.

    Correspondingly, `vector_field` and `control` should both return PyTrees, and both
    should have the same structure and leaf shape as the initial state `y0`. These are
    multiplied together pointwise.

    !!! info

        Why "weakly" diagonal? Consider the matrix representation of the vector field,
        as a square diagonal matrix. In general, the (i,i)-th element may depending
        upon any of the values of `y`. It is only if the (i,i)-th element only depends
        upon the i-th element of `y` that the vector field is said to be "diagonal",
        without the "weak". (This stronger property is useful in some SDE solvers.)
    """

    def __check_init__(self):
        warnings.warn(
            "`WeaklyDiagonalControlTerm` is now deprecated, in favour combining "
            "`ControlTerm` with a `lineax.AbstractLinearOperator`. This offers a way "
            "to define a vector field with any kind of structure -- diagonal or "
            "otherwise.\n"
            "For a diagonal linear operator, then this can be easily converted as "
            "follows. What was previously:\n"
            "```\n"
            "def vector_field(t, y, args):\n"
            "    ...\n"
            "    return some_vector\n"
            "\n"
            "diffrax.WeaklyDiagonalControlTerm(vector_field)\n"
            "```\n"
            "is now:\n"
            "```\n"
            "import lineax\n"
            "\n"
            "def vector_field(t, y, args):\n"
            "    ...\n"
            "    return lineax.DiagonalLinearOperator(some_vector)\n"
            "\n"
            "diffrax.ControlTerm(vector_field)\n"
            "```\n"
            "Lineax is available at `https://github.com/patrick-kidger/lineax`.\n",
            stacklevel=3,
        )

    def prod(self, vf: _VF, control: _Control) -> Y:
        with jax.numpy_dtype_promotion("standard"):
            return jtu.tree_map(operator.mul, vf, control)

class AdjointTerm(AbstractTerm[_VF, _Control]):
    term: AbstractTerm[_VF, _Control]

    def is_vf_expensive(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y: tuple[
            PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
        ],
        args: Args,
    ) -> bool:
        control_struct = eqx.filter_eval_shape(self.contr, t0, t1)
        if sum(c.size for c in jtu.tree_leaves(control_struct)) in (0, 1):
            return False
        else:
            return True

    def vf(
        self,
        t: RealScalarLike,
        y: tuple[
            PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
        ],
        args: Args,
    ) -> PyTree[ArrayLike]:
        # We compute the vector field via `self.vf_prod`. We could also do it manually,
        # but this is relatively painless.
        #
        # This can be done because `self.vf_prod` is linear in `control`. As such we
        # can obtain just the vector field component by representing this linear
        # operation as a matrix. Which in turn is simply computing the Jacobian.
        #
        # Notes:
        # - Whilst `self.vf_prod` also involves autodifferentiation, we don't
        #   actually compute a second derivative anywhere. (The derivatives are of
        #   different quantities.)
        # - Because the operation is linear, then in some sense this Jacobian isn't
        #   really doing any autodifferentiation at all.
        # - If we wanted we could manually perform the operations that this Jacobian is
        #   doing; in particular this requires `jax.linear_transpose`-ing
        #   `self.term.prod` to get something `control`-shaped.

        # The value of `control` is never actually used -- just its shape, dtype, and
        # PyTree structure. (This is because `self.vf_prod` is linear in `control`.)
        control = self.contr(t, t)

        y_size = sum(np.size(yi) for yi in jtu.tree_leaves(y))
        control_size = sum(np.size(ci) for ci in jtu.tree_leaves(control))
        if y_size > control_size:
            make_jac = jax.jacfwd
        else:
            make_jac = jax.jacrev

        # Find the tree structure of vf_prod by smuggling it out as an additional
        # result from the Jacobian calculation.
        sentinel = vf_prod_tree = object()
        control_tree = jtu.tree_structure(control)

        def _fn(_control):
            _out = self.vf_prod(t, y, args, _control)
            nonlocal vf_prod_tree
            structure = jtu.tree_structure(_out)
            if vf_prod_tree is sentinel:
                vf_prod_tree = structure
            else:
                assert vf_prod_tree == structure
            return _out

        jac = make_jac(_fn)(control)
        assert vf_prod_tree is not sentinel
        vf_prod_tree = cast(PyTreeDef, vf_prod_tree)
        if jtu.tree_structure(None) in (vf_prod_tree, control_tree):
            # An unusual/not-useful edge case to handle.
            raise NotImplementedError(
                "`AdjointTerm.vf` not implemented for `None` controls or states."
            )
        return jtu.tree_transpose(vf_prod_tree, control_tree, jac)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        return self.term.contr(t0, t1, **kwargs)

    def prod(
        self, vf: PyTree[ArrayLike], control: _Control
    ) -> tuple[
        PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
    ]:
        # As per what is returned from `self.vf`, then `vf` has a PyTree structure of
        # (control_tree, vf_prod_tree)

        # Calculate vf_prod_tree by smuggling it out.
        sentinel = vf_prod_tree = object()
        control_tree = jtu.tree_structure(control)

        def _get_vf_tree(_, tree):
            nonlocal vf_prod_tree
            structure = jtu.tree_structure(tree)
            if vf_prod_tree is sentinel:
                vf_prod_tree = structure
            else:
                assert vf_prod_tree == structure

        jtu.tree_map(_get_vf_tree, control, vf)
        assert vf_prod_tree is not sentinel
        vf_prod_tree = cast(PyTreeDef, vf_prod_tree)

        vf = jtu.tree_transpose(control_tree, vf_prod_tree, vf)

        example_vf_prod = jtu.tree_unflatten(
            vf_prod_tree, [0 for _ in range(vf_prod_tree.num_leaves)]
        )

        def _contract(_, vf_piece):
            assert jtu.tree_structure(vf_piece) == control_tree
            _contracted = jtu.tree_map(_prod, vf_piece, control)
            return sum(jtu.tree_leaves(_contracted), 0)

        return jtu.tree_map(_contract, example_vf_prod, vf)

    def vf_prod(
        self,
        t: RealScalarLike,
        y: tuple[
            PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
        ],
        args: Args,
        control: _Control,
    ) -> tuple[
        PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
    ]:
        # Note the inclusion of "implicit" parameters (as `term` might be a callable
        # PyTree a la Equinox) and "explicit" parameters (`args`)
        y, a_y, _, _ = y
        diff_args, nondiff_args = eqx.partition(args, eqx.is_inexact_array)
        diff_term, nondiff_term = eqx.partition(self.term, eqx.is_inexact_array)

        def _to_vjp(_y, _diff_args, _diff_term):
            _args = eqx.combine(_diff_args, nondiff_args)
            _term = eqx.combine(_diff_term, nondiff_term)
            return _term.vf_prod(t, _y, _args, control)

        dy, vjp = jax.vjp(_to_vjp, y, diff_args, diff_term)
        da_y, da_diff_args, da_diff_term = vjp((-(a_y**ω)).ω)
        return dy, da_y, da_diff_args, da_diff_term

