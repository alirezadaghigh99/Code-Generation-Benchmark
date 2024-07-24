class KineticEnergy(ContinuousOperator):
    r"""This is the kinetic energy operator (hbar = 1). The local value is given by:
    :math:`E_{kin} = -1/2 ( \sum_i \frac{1}{m_i} (\log(\psi))'^2 + (\log(\psi))'' )`
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        mass: Union[float, list[float]],
        dtype: Optional[DType] = None,
    ):
        r"""Args:
        hilbert: The underlying Hilbert space on which the operator is defined
        mass: float if all masses are the same, list indicating the mass of each particle otherwise
        dtype: Data type of the mass
        """

        self._mass = jnp.asarray(mass, dtype=dtype)

        self._is_hermitian = np.allclose(self._mass.imag, 0.0)
        self.__attrs = None

        super().__init__(hilbert, self._mass.dtype)

    @property
    def mass(self):
        return self._mass

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _expect_kernel_single(
        self, logpsi: Callable, params: PyTree, x: Array, inverse_mass: Optional[PyTree]
    ):
        def logpsi_x(x):
            return logpsi(params, x)

        dlogpsi_x = jacrev(logpsi_x)

        dp_dx2 = jnp.diag(jacfwd(dlogpsi_x)(x)[0].reshape(x.shape[0], x.shape[0]))
        dp_dx = dlogpsi_x(x)[0][0] ** 2

        return -0.5 * jnp.sum(inverse_mass * (dp_dx2 + dp_dx), axis=-1)

    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, coefficient: Optional[PyTree]
    ):
        return self._expect_kernel_single(logpsi, params, x, coefficient)

    def _pack_arguments(self) -> PyTree:
        return 1.0 / self._mass

    @property
    def _attrs(self):
        if self.__attrs is None:
            self.__attrs = (self.hilbert, self.dtype, HashableArray(self.mass))
        return self.__attrs

    def __repr__(self):
        return f"KineticEnergy(m={self._mass})"

