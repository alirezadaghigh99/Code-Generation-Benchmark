def random_state(
        self,
        key=None,
        size: Optional[int] = None,
        dtype=np.float32,
    ) -> jnp.ndarray:
        r"""Generates either a single or a batch of uniformly distributed random states.
        Runs as :code:`random_state(self, key, size=None, dtype=np.float32)` by default.

        Args:
            key: rng state from a jax-style functional generator.
            size: If provided, returns a batch of configurations of the form
                  :code:`(size, N)` if size is an integer or :code:`(*size, N)` if it is
                  a tuple and where :math:`N` is the Hilbert space size.
                  By default, a single random configuration with shape
                  :code:`(#,)` is returned.
            dtype: DType of the resulting vector.

        Returns:
            A state or batch of states sampled from the uniform distribution on the
            hilbert space.

        Example:

            >>> import netket, jax
            >>> hi = netket.hilbert.Qubit(N=2)
            >>> k1, k2 = jax.random.split(jax.random.PRNGKey(1))
            >>> print(hi.random_state(key=k1))
            [1. 0.]
            >>> print(hi.random_state(key=k2, size=2))
            [[0. 0.]
             [0. 1.]]
        """
        from netket.hilbert import random

        return random.random_state(self, key, size, dtype=dtype)

