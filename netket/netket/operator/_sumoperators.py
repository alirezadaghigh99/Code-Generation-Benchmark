class SumOperator(ContinuousOperator):
    r"""This class implements the action of the _expect_kernel()-method of
    ContinuousOperator for a sum of ContinuousOperator objects.
    """

    def __init__(
        self,
        *operators: tuple[ContinuousOperator, ...],
        coefficients: Union[float, Iterable[float]] = 1.0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Returns the action of a sum of local operators.
        Args:
            operators: A list of ContinuousOperator objects
            coefficients: A coefficient for each ContinuousOperator object
            dtype: Data type of the coefficients
        """
        hi_spaces = [op.hilbert for op in operators]
        if not all(hi == hi_spaces[0] for hi in hi_spaces):
            raise NotImplementedError(
                "Cannot add operators on different hilbert spaces"
            )

        if is_scalar(coefficients):
            coefficients = [coefficients for _ in operators]

        if len(operators) != len(coefficients):
            raise AssertionError("Each operator needs a coefficient")

        operators, coefficients = _flatten_sumoperators(operators, coefficients)

        dtype = canonicalize_dtypes(float, *operators, *coefficients, dtype=dtype)

        self._operators = tuple(operators)
        self._coefficients = jnp.asarray(coefficients, dtype=dtype)

        super().__init__(hi_spaces[0], self._coefficients.dtype)

        self._is_hermitian = all([op.is_hermitian for op in operators])
        self.__attrs = None

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    @property
    def operators(self) -> tuple[ContinuousOperator, ...]:
        """The list of all operators in the terms of this sum. Every
        operator is summed with a corresponding coefficient
        """
        return self._operators

    @property
    def coefficients(self) -> Array:
        return self._coefficients

    @staticmethod
    def _expect_kernel(
        logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        result = [
            data.coeffs[i] * op._expect_kernel(logpsi, params, x, op_data)
            for i, (op, op_data) in enumerate(zip(data.ops, data.op_data))
        ]

        return sum(result)

    def _pack_arguments(self) -> SumOperatorPyTree:
        return SumOperatorPyTree(
            self.operators,
            self.coefficients,
            tuple(op._pack_arguments() for op in self.operators),
        )

    @property
    def _attrs(self) -> tuple[Hashable, ...]:
        if self.__attrs is None:
            self.__attrs = (
                self.hilbert,
                self.operators,
                HashableArray(self.coefficients),
                self.dtype,
            )
        return self.__attrs

    def __repr__(self):
        return (
            f"SumOperator(operators={self.operators}, coefficients={self.coefficients})"
        )

