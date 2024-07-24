class PotentialEnergy(ContinuousOperator):
    r"""Returns the local potential energy defined in afun"""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        afun: Callable,
        coefficient: float = 1.0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            afun: The potential energy as function of x
            coefficient: A coefficient for the ContinuousOperator object
            dtype: Data type of the coefficient
        """

        self._afun = afun
        self._coefficient = jnp.array(coefficient, dtype=dtype)

        self.__attrs = None

        super().__init__(hilbert, self._coefficient.dtype)

    @property
    def coefficient(self) -> Array:
        return self._coefficient

    @property
    def is_hermitian(self) -> bool:
        return True

    @staticmethod
    def _expect_kernel(
        logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ) -> Array:
        return data.coefficient * jax.vmap(data.potential_fun, in_axes=(0,))(x)

    def _pack_arguments(self) -> PotentialOperatorPyTree:
        return PotentialOperatorPyTree(self._afun, self.coefficient)

    @property
    def _attrs(self) -> tuple[Hashable, ...]:
        if self.__attrs is None:
            self.__attrs = (
                self.hilbert,
                self._afun,
                self.dtype,
                HashableArray(self.coefficient),
            )
        return self.__attrs

    def __repr__(self):
        return f"Potential(coefficient={self.coefficient}, function={self._afun})"

