class FermionOperator2nd(FermionOperator2ndBase):
    r"""
    A fermionic operator in :math:`2^{nd}` quantization, using Numba
    for indexing.

    .. warning::

        This class is not a Pytree, so it cannot be used inside of jax-transformed
        functions like `jax.grad` or `jax.jit`.

        The standard usage is to index into the operator from outside the jax
        function transformation and pass the results to the jax-transformed functions.

        To use this operator inside of a jax function transformation, convert it to a
        jax operator (class:`netket.experimental.operator.FermionOperator2ndJax`) by using
        :meth:`netket.experimental.operator.FermionOperator2nd.to_jax_operator()`.

    When using native (experimental) sharding, or when working with GPUs,
    we reccomend using the Jax implementations of the operators for potentially
    better performance.
    """

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:
            # following lists will be used to compute matrix elements
            # they are filled in _add_term
            out = _pack_internals(self._operators, self._dtype)
            (
                self._orb_idxs,
                self._daggers,
                self._numba_weights,
                self._diag_idxs,
                self._off_diag_idxs,
                self._term_split_idxs,
            ) = out

            self._max_conn_size = 0
            if len(self._diag_idxs) > 0:
                self._max_conn_size += 1
            # the following could be reduced further
            self._max_conn_size += len(self._off_diag_idxs)

            self._initialized = True

    def to_jax_operator(self) -> "FermionOperator2ndJax":  # noqa: F821
        """
        Returns the jax version of this operator, which is an
        instance of :class:`netket.experimental.operator.FermionOperator2ndJax`.
        """
        from ._fermion_operator_2nd_jax import FermionOperator2ndJax

        new_op = FermionOperator2ndJax(
            self.hilbert, cutoff=self._cutoff, dtype=self.dtype
        )
        new_op._operators = self._operators.copy()
        return new_op

    def _get_conn_flattened_closure(self):
        self._setup()
        _max_conn_size = self.max_conn_size
        _orb_idxs = self._orb_idxs
        _daggers = self._daggers
        _weights = self._numba_weights
        _diag_idxs = self._diag_idxs
        _off_diag_idxs = self._off_diag_idxs
        _term_split_idxs = self._term_split_idxs
        _cutoff = self._cutoff

        fun = self._flattened_kernel

        def gccf_fun(x, sections):
            return fun(
                x,
                sections,
                _max_conn_size,
                _orb_idxs,
                _daggers,
                _weights,
                _diag_idxs,
                _off_diag_idxs,
                _term_split_idxs,
                _cutoff,
            )

        return numba.jit(nopython=True)(gccf_fun)

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator.

        Starting from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x: A matrix of shape (batch_size,hilbert.size) containing
                the batch of quantum numbers x.
            sections: An array of size (batch_size) useful to unflatten
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
            self.max_conn_size,
            self._orb_idxs,
            self._daggers,
            self._numba_weights,
            self._diag_idxs,
            self._off_diag_idxs,
            self._term_split_idxs,
            self._cutoff,
            pad,
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _flattened_kernel(  # pragma: no cover
        x,
        sections,
        max_conn,
        orb_idxs,
        daggers,
        weights,
        diag_idxs,
        off_diag_idxs,
        term_split_idxs,
        cutoff,
        pad=False,
    ):
        x_prime = np.empty((x.shape[0] * max_conn, x.shape[1]), dtype=x.dtype)
        mels = np.zeros((x.shape[0] * max_conn), dtype=weights.dtype)

        # do not split at the last one (gives empty array)
        term_split_idxs = term_split_idxs[:-1]
        orb_idxs_list = np.split(orb_idxs, term_split_idxs)
        daggers_list = np.split(daggers, term_split_idxs)

        # loop over the batch dimension
        n_c = 0
        for b in range(x.shape[0]):
            xb = x[b, :]

            # we can already fill up with default values
            if pad:
                x_prime[b * max_conn : (b + 1) * max_conn, :] = np.copy(xb)

            non_zero_diag = False
            # first do the diagonal terms, they all generate just 1 term
            for term_idx in diag_idxs:
                mel = weights[term_idx]
                xt = np.copy(xb)
                has_xp = True
                for orb_idx, dagger in zip(
                    orb_idxs_list[term_idx], daggers_list[term_idx]
                ):
                    _, mel, op_has_xp = _apply_operator(
                        xt, orb_idx, dagger, mel, cutoff
                    )
                    if not op_has_xp:
                        has_xp = False
                        continue
                if has_xp:
                    x_prime[n_c, :] = np.copy(xb)  # should be untouched
                    mels[n_c] += mel

                non_zero_diag = non_zero_diag or has_xp

            # end of the diagonal terms
            if non_zero_diag:
                n_c += 1

            # now do the off-diagonal terms
            for term_idx in off_diag_idxs:
                mel = weights[term_idx]
                xt = np.copy(xb)
                has_xp = True
                for orb_idx, dagger in zip(
                    orb_idxs_list[term_idx], daggers_list[term_idx]
                ):
                    xt, mel, op_has_xp = _apply_operator(
                        xt, orb_idx, dagger, mel, cutoff
                    )
                    if not op_has_xp:  # detect zeros
                        has_xp = False
                        continue
                if has_xp:
                    x_prime[n_c, :] = np.copy(xt)  # should be different
                    mels[n_c] += mel
                    n_c += 1

            # end of this sample
            if pad:
                n_c = (b + 1) * max_conn

            sections[b] = n_c

        if pad:
            return x_prime, mels
        else:
            return x_prime[:n_c], mels[:n_c]

