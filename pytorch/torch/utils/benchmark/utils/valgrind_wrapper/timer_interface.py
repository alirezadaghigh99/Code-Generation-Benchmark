class FunctionCounts:
    """Container for manipulating Callgrind results.

    It supports:
        1) Addition and subtraction to combine or diff results.
        2) Tuple-like indexing.
        3) A `denoise` function which strips CPython calls which are known to
           be non-deterministic and quite noisy.
        4) Two higher order methods (`filter` and `transform`) for custom
           manipulation.
    """
    _data: Tuple[FunctionCount, ...]
    inclusive: bool
    truncate_rows: bool = True

    # For normal use, torch._tensor_str.PRINT_OPTS.linewidth determines
    # the print settings. This is simply to allow hermetic unit tests.
    _linewidth: Optional[int] = None

    def __iter__(self) -> Iterator[FunctionCount]:
        yield from self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Any) -> Union[FunctionCount, "FunctionCounts"]:
        data: Union[FunctionCount, Tuple[FunctionCount, ...]] = self._data[item]
        return (
            FunctionCounts(cast(Tuple[FunctionCount, ...], data), self.inclusive, truncate_rows=False)
            if isinstance(data, tuple) else data
        )

    def __repr__(self) -> str:
        count_len = 0
        for c, _ in self:
            # Account for sign in string length.
            count_len = max(count_len, len(str(c)) + int(c < 0))

        lines = []
        linewidth = self._linewidth or torch._tensor_str.PRINT_OPTS.linewidth
        fn_str_len = max(linewidth - count_len - 4, 40)
        for c, fn in self:
            if len(fn) > fn_str_len:
                left_len = int((fn_str_len - 5) // 2)
                fn = fn[:left_len] + " ... " + fn[-(fn_str_len - left_len - 5):]
            lines.append(f"  {c:>{count_len}}  {fn}")

        if self.truncate_rows and len(lines) > 18:
            lines = lines[:9] + ["...".rjust(count_len + 2)] + lines[-9:]

        if not self.inclusive:
            lines.extend(["", f"Total: {self.sum()}"])

        return "\n".join([super().__repr__()] + lines)

    def __add__(
        self,
        other: "FunctionCounts",
    ) -> "FunctionCounts":
        return self._merge(other, lambda c: c)

    def __sub__(
        self,
        other: "FunctionCounts",
    ) -> "FunctionCounts":
        return self._merge(other, operator.neg)

    def __mul__(self, other: Union[int, float]) -> "FunctionCounts":
        return self._from_dict({
            fn: int(c * other) for c, fn in self._data
        }, self.inclusive)

    def transform(self, map_fn: Callable[[str], str]) -> "FunctionCounts":
        """Apply `map_fn` to all of the function names.

        This can be used to regularize function names (e.g. stripping irrelevant
        parts of the file path), coalesce entries by mapping multiple functions
        to the same name (in which case the counts are added together), etc.
        """
        counts: DefaultDict[str, int] = collections.defaultdict(int)
        for c, fn in self._data:
            counts[map_fn(fn)] += c

        return self._from_dict(counts, self.inclusive)

    def filter(self, filter_fn: Callable[[str], bool]) -> "FunctionCounts":
        """Keep only the elements where `filter_fn` applied to function name returns True."""
        return FunctionCounts(tuple(i for i in self if filter_fn(i.function)), self.inclusive)

    def sum(self) -> int:
        return sum(c for c, _ in self)

    def denoise(self) -> "FunctionCounts":
        """Remove known noisy instructions.

        Several instructions in the CPython interpreter are rather noisy. These
        instructions involve unicode to dictionary lookups which Python uses to
        map variable names. FunctionCounts is generally a content agnostic
        container, however this is sufficiently important for obtaining
        reliable results to warrant an exception."""
        return self.filter(lambda fn: "dictobject.c:lookdict_unicode" not in fn)

    def _merge(
        self,
        second: "FunctionCounts",
        merge_fn: Callable[[int], int]
    ) -> "FunctionCounts":
        assert self.inclusive == second.inclusive, "Cannot merge inclusive and exclusive counts."
        counts: DefaultDict[str, int] = collections.defaultdict(int)
        for c, fn in self:
            counts[fn] += c

        for c, fn in second:
            counts[fn] += merge_fn(c)

        return self._from_dict(counts, self.inclusive)

    @staticmethod
    def _from_dict(counts: Dict[str, int], inclusive: bool) -> "FunctionCounts":
        flat_counts = (FunctionCount(c, fn) for fn, c in counts.items() if c)
        return FunctionCounts(tuple(sorted(flat_counts, reverse=True)), inclusive)

class FunctionCount(NamedTuple):
    # TODO(#105471): Rename the count field
    count: int  # type: ignore[assignment]
    function: str

class CopyIfCallgrind:
    """Signal that a global may be replaced with a deserialized copy.

    See `GlobalsBridge` for why this matters.
    """
    def __init__(self, value: Any, *, setup: Optional[str] = None):
        for method, supported_types in _GLOBALS_ALLOWED_TYPES.items():
            if any(isinstance(value, t) for t in supported_types):
                self._value: Any = value
                self._setup: Optional[str] = setup
                self._serialization: Serialization = method
                break
        else:
            supported_str = "\n".join([
                getattr(t, "__name__", repr(t))
                for t in it.chain(_GLOBALS_ALLOWED_TYPES.values())])

            raise ValueError(
                f"Unsupported type: {type(value)}\n"
                f"`collect_callgrind` restricts globals to the following types:\n"
                f"{textwrap.indent(supported_str, '  ')}"
            )

    @property
    def value(self) -> Any:
        return self._value

    @property
    def setup(self) -> Optional[str]:
        return self._setup

    @property
    def serialization(self) -> Serialization:
        return self._serialization

    @staticmethod
    def unwrap_all(globals: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: (v.value if isinstance(v, CopyIfCallgrind) else v)
            for k, v in globals.items()
        }

