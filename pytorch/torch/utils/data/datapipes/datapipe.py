class DataChunk(List[_T]):
    def __init__(self, items: Iterable[_T]) -> None:
        items = list(items)
        super().__init__(items)
        self.items = items

    def as_str(self, indent: str = "") -> str:
        return indent + "[" + ", ".join(str(i) for i in iter(self)) + "]"

    def __iter__(self) -> Iterator[_T]:
        yield from super().__iter__()

    def raw_iterator(self) -> Iterator[_T]:
        yield from self.items

