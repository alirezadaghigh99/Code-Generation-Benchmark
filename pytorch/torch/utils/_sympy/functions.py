class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)

    NB: This is Python-style floor division, round to -Inf
    """

    nargs = (2,)
    precedence = 50  # precedence of mul  # noqa: F811

    is_integer = True

    @property
    def base(self):
        return self.args[0]

    @property
    def divisor(self):
        return self.args[1]

    def _sympystr(self, printer):
        base = printer.parenthesize(self.base, self.precedence)
        divisor = printer.parenthesize(self.divisor, self.precedence)
        return f"({base}//{divisor})"

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base, divisor):
        # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
        # Assert triggered by inequality solver
        # assert base.is_integer, base
        # assert divisor.is_integer, divisor

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        if base in (int_oo, -int_oo, sympy.oo, -sympy.oo) and divisor in (
            int_oo,
            -int_oo,
            sympy.oo,
            -sympy.oo,
        ):
            return sympy.nan
        if base is sympy.nan or divisor is sympy.nan:
            return sympy.nan

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and divisor == 1:
            return base
        if base.is_integer and divisor == -1:
            return sympy.Mul(base, -1)
        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (
                base in (int_oo, -int_oo, sympy.oo, -sympy.oo)
                or divisor in (int_oo, -int_oo, sympy.oo, -sympy.oo)
            )
        ):
            r = float(base) / float(divisor)
            if r == math.inf:
                return int_oo
            elif r == -math.inf:
                return -int_oo
            elif math.isnan(r):
                return sympy.nan
            else:
                return sympy.Integer(math.floor(r))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Integer(int(base) // int(divisor))
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        # Expands (x + y) // b into x // b + y // b.
        # This only works if floor is an identity, i.e. x / b is an integer.
        for term in sympy.Add.make_args(base):
            quotient = term / divisor
            if quotient.is_integer and isinstance(divisor, sympy.Integer):
                # NB: this is correct even if the divisor is not an integer, but it
                # creates rational expressions that cause problems with dynamic
                # shapes.
                return FloorDiv(base - term, divisor) + quotient

        try:
            gcd = sympy.gcd(base, divisor)
            if gcd != 1:
                return FloorDiv(
                    sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
                )
        except sympy.PolynomialError:
            pass  # https://github.com/pytorch/pytorch/issues/108276

