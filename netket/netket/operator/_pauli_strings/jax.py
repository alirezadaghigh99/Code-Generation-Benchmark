class PauliStringsJax(PauliStringsBase, DiscreteJaxOperator):
    """
    Jax-compatible version of :class:`netket.operator.PauliStrings`.
    """

    @wraps(PauliStringsBase.__init__)
    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: Union[None, str, list[str]] = None,
        weights: Union[None, float, complex, list[Union[float, complex]]] = None,
        *,
        cutoff: float = 1.0e-10,
        dtype: Optional[DType] = None,
        _mode: str = "index",
    ):
        super().__init__(hilbert, operators, weights, cutoff=cutoff, dtype=dtype)

        if len(self.hilbert.local_states) != 2:
            raise ValueError(
                "PauliStringsJax only supports Hamiltonians with two local states"
            )

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

        # private variable for setting the mode
        # currently there are two modes:
        # index: indexes into the vector to flip qubits and compute the sign
        #           faster if the strings act only on a few qubits
        # mask: uses masks to flip qubits and compute the sign
        #          faster if the strings act on many of the qubits
        #          (and possibly on gpu)
        # By adapting pack_internals_jax hybrid approaches are also possible.
        # depending on performance tests we might expose or remove it
        self._hi_local_states = tuple(self.hilbert.local_states)
        self._initialized = False
        self._mode = _mode

    @property
    def _mode(self):
        """
        (Internal) Indexing mode of the operator.

        Valid values are "index" or "mask".

        'Index' uses the standard LocalOperator-like indexing of changed points,
        while the latter uses constant-size masks.

        The latter does not really need recompilation for paulistrings with
        different values, and this could be changed in the future.
        """
        return self._mode_attr

    @_mode.setter
    def _mode(self, mode):
        _check_mode(mode)
        self._mode_attr = mode
        self._reset_caches()

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        self._setup()
        return self._x_flip_masks_stacked.shape[0]

    def _setup(self, force=False):
        if force or not self._initialized:
            weights = concrete_or_error(
                np.asarray,
                self.weights,
                JaxOperatorSetupDuringTracingError,
                self,
            )

            # Necessary for the tree_flatten in jax.jit, because
            # metadata must be hashable and comparable. We don't
            # want to re-hash it at every unpacking so we do it
            # once in here.
            if self._mode == "index":
                self._operators_hashable = HashableArray(self.operators)
            else:
                self._operators_hashable = None

            x_flip_masks_stacked, z_data = pack_internals_jax(
                self.operators, weights, weight_dtype=self.dtype, mode=self._mode
            )
            self._x_flip_masks_stacked = x_flip_masks_stacked
            self._z_data = z_data
            self._initialized = True

    def _reset_caches(self):
        super()._reset_caches()
        self._initialized = False

    def n_conn(self, x):
        self._setup()
        return _pauli_strings_n_conn_jax(
            self._hi_local_states,
            self._x_flip_masks_stacked,
            self._z_data,
            x,
            cutoff=self._cutoff,
        )

    def get_conn_padded(self, x):
        self._setup()
        xp, mels, _ = _pauli_strings_kernel_jax(
            self._hi_local_states,
            self._x_flip_masks_stacked,
            self._z_data,
            x,
            cutoff=self._cutoff,
        )
        return xp, mels

    def tree_flatten(self):
        self._setup()
        data = (self.weights, self._x_flip_masks_stacked, self._z_data)
        metadata = {
            "hilbert": self.hilbert,
            "operators": self._operators_hashable,
            "dtype": self.dtype,
            "mode": self._mode,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        (weights, xm, zd) = data
        hi = metadata["hilbert"]
        operators_hashable = metadata["operators"]
        dtype = metadata["dtype"]
        mode = metadata["mode"]

        op = cls(hi, dtype=dtype, _mode=mode)
        op._operators = (
            operators_hashable.wrapped if operators_hashable is not None else None
        )
        op._operators_hashable = operators_hashable
        op._weights = weights
        op._x_flip_masks_stacked = xm
        op._z_data = zd
        op._initialized = True
        return op

    def to_numba_operator(self) -> "PauliStrings":  # noqa: F821
        """
        Returns the standard numba version of this operator, which is an
        instance of :class:`netket.operator.PauliStrings`.
        """
        from .numba import PauliStrings

        return PauliStrings(
            self.hilbert,
            self.operators,
            self.weights,
            dtype=self.dtype,
            cutoff=self._cutoff,
        )

