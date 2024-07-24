class Sum(ReducingCombination):
    @property
    def _reduce(self) -> Callable[[Sequence[TensorType]], TensorType]:
        return tf.add_n  # type: ignore[no-any-return]

