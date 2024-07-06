def numbers_to_states(self, numbers: Array) -> Array:
        r"""Returns the quantum numbers corresponding to the n-th basis state
        for input n.

        `n` is an array of integer indices such that
        :code:`numbers[k]=Index(states[k])`.
        Throws an exception iff the space is not indexable.

        This function validates the inputs by checking that the numbers provided
        are smaller than the Hilbert space size, and throws an error if that
        condition is not met. When called from within a `jax.jit` context, this
        uses {func}`equinox.error_if` to throw runtime errors.

        Args:
            numbers (numpy.array): Batch of input numbers to be converted into arrays of
                quantum numbers.
        """

        if not self.is_indexable:
            raise RuntimeError("The hilbert space is too large to be indexed.")

        numbers = jnp.asarray(numbers, dtype=np.int32)

        numbers = error_if(
            numbers,
            (numbers >= self.n_states).any() | (numbers < 0).any(),
            "Numbers outside the range of allowed states.",
        )

        return self._numbers_to_states(numbers.ravel()).reshape(
            (*numbers.shape, self.size)
        )

