    def values(cls) -> FrozenSet[str]:
        """Returns all values."""
        return frozenset(arg.value for arg in cls)