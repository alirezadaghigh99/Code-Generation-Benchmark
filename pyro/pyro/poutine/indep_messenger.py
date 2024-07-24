class CondIndepStackFrame(NamedTuple):
    name: str
    dim: Optional[int]
    size: int
    counter: int
    full_size: Optional[int] = None

    @property
    def vectorized(self) -> bool:
        return self.dim is not None

    def _key(self) -> Tuple[str, Optional[int], int, int]:
        size = self.size
        with ignore_jit_warnings(["Converting a tensor to a Python number"]):
            if isinstance(size, torch.Tensor):  # type: ignore[unreachable]
                size = size.item()  # type: ignore[unreachable]
        return self.name, self.dim, size, self.counter

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CondIndepStackFrame):
            return False
        return self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._key())

    def __str__(self) -> str:
        return self.name

