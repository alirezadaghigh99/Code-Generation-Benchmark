class PauliStrings(PauliStringsBase):
    """A Hamiltonian consisting of the sum of products of Pauli operators."""

    @wraps(PauliStringsBase.__init__)
    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: Union[None, str, list[str]] = None,
        weights: Union[None, float, complex, list[Union[float, complex]]] = None,
        *,
        cutoff: float = 1.0e-10,
        dtype: Optional[DType] = None,
    ):
        super().__init__(hilbert, operators, weights, cutoff=cutoff, dtype=dtype)

        # check that it is homogeneous, throw error if it's not
        if not isinstance(self.hilbert, HomogeneousHilbert):
            local_states = self.hilbert.states_at_index(0)
            if not all(
                np.allclose(local_states, self.hilbert.states_at_index(i))
                for i in range(self.hilbert.size)
            ):
                raise ValueError(
                    "Hilbert spaces with non homogeneous local_states are not "
                    "yet supported by PauliStrings."
                )

        self._initialized = False

    def to_jax_operator(self) -> "PauliStringsJax":  # noqa: F821
        """
        Returns the jax-compatible version of this operator, which is an
        instance of :class:`netket.operator.PauliStringsJax`.
        """
        from .jax import PauliStringsJax

        return PauliStringsJax(
            self.hilbert,
            self.operators,
            self.weights,
            dtype=self.dtype,
            cutoff=self._cutoff,
        )

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        # 1 connection for every operator X, Y, Z...
        self._setup()
        return self._n_operators

    def _setup(self, force=False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:
            data = pack_internals_numba(
                self.hilbert, self.operators, self.weights, self.dtype, self._cutoff
            )

            self._sites = data["sites"]
            self._ns = data["ns"]
            self._n_op = data["n_op"]
            self._weights_numba = data["weights_numba"]
            self._nz_check = data["nz_check"]
            self._z_check = data["z_check"]
            self._n_operators = data["n_operators"]

            # caches for execution
            self._x_prime_max = np.empty((self._n_operators, self.hilbert.size))
            self._mels_max = np.empty((self._n_operators), dtype=data["mel_dtype"])

            self._local_states = np.array(self.hilbert.states_at_index(0))
            self._initialized = True

    def _reset_caches(self):
        super()._reset_caches()
        self._initialized = False

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
        x,
        sections,
        x_prime,
        mels,
        sites,
        ns,
        n_op,
        weights,
        nz_check,
        z_check,
        cutoff,
        max_conn,
        local_states,
        pad=False,
    ):
        x_prime = np.empty((x.shape[0] * max_conn, x_prime.shape[1]), dtype=x.dtype)
        mels = np.zeros((x.shape[0] * max_conn), dtype=mels.dtype)
        state_1 = local_states[-1]

        n_c = 0
        for b in range(x.shape[0]):
            xb = x[b]
            # initialize with the old state
            x_prime[b * max_conn : (b + 1) * max_conn, :] = np.copy(xb)

            for i in range(sites.shape[0]):  # iterate over the X strings
                mel = 0.0
                # iterate over the Z substrings
                for j in range(n_op[i]):
                    # apply all the Z (check the qubits at all affected sites)
                    if nz_check[i, j] > 0:
                        to_check = z_check[i, j, : nz_check[i, j]]
                        n_z = np.count_nonzero(xb[to_check] == state_1)
                    else:
                        n_z = 0
                    # multiply with -1 for every site we did Z which was state_1
                    mel += weights[i, j] * (-1.0) ** n_z

                if cutoff is None or abs(mel) > cutoff:
                    x_prime[n_c] = np.copy(xb)
                    # now flip all the sites in the X string
                    for site in sites[i, : ns[i]]:
                        new_state_idx = int(x_prime[n_c, site] == local_states[0])
                        x_prime[n_c, site] = local_states[new_state_idx]
                    mels[n_c] = mel
                    n_c += 1

            if pad:
                n_c = (b + 1) * max_conn

            sections[b] = n_c
        return x_prime[:n_c], mels[:n_c]

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        self._setup()

        x = concrete_or_error(
            np.asarray,
            x,
            NumbaOperatorGetConnDuringTracingError,
            self,
        )

        assert (
            x.shape[-1] == self.hilbert.size
        ), "size of hilbert space does not match size of x"
        return self._flattened_kernel(
            x,
            sections,
            self._x_prime_max,
            self._mels_max,
            self._sites,
            self._ns,
            self._n_op,
            self._weights_numba,
            self._nz_check,
            self._z_check,
            self._cutoff,
            self._n_operators,
            self._local_states,
            pad,
        )

    def _get_conn_flattened_closure(self):
        self._setup()
        _x_prime_max = self._x_prime_max
        _mels_max = self._mels_max
        _sites = self._sites
        _ns = self._ns
        _n_op = self._n_op
        _weights = self._weights_numba
        _nz_check = self._nz_check
        _z_check = self._z_check
        _cutoff = self._cutoff
        _n_operators = self._n_operators
        fun = self._flattened_kernel
        _local_states = self._local_states

        def gccf_fun(x, sections):
            return fun(
                x,
                sections,
                _x_prime_max,
                _mels_max,
                _sites,
                _ns,
                _n_op,
                _weights,
                _nz_check,
                _z_check,
                _cutoff,
                _n_operators,
                _local_states,
            )

        return jit(nopython=True)(gccf_fun)