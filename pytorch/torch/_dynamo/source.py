class GetItemSource(ChainedSource):
    index: Any
    index_is_slice: bool = False

    def __post_init__(self):
        assert self.base is not None
        if isinstance(self.index, slice):
            # store the hashable version of the slice so the whole GetItemSource is hashable
            super().__setattr__("index", self.index.__reduce__())
            super().__setattr__("index_is_slice", True)

    def reconstruct(self, codegen):
        reconstruct_getitem(self, codegen, index_is_slice=self.index_is_slice)
        codegen.append_output(create_instruction("BINARY_SUBSCR"))

    def guard_source(self):
        return self.base.guard_source()

    def unpack_slice(self):
        assert self.index_is_slice
        slice_class, slice_args = self.index
        return slice_class(*slice_args)

    def name(self):
        # Index can be of following types
        # 1) ConstDictKeySource
        # 2) enum.Enum
        # 3) index is a slice - example 1:4
        # 4) index is a constant - example string, integer
        if isinstance(self.index, Source):
            if not isinstance(self.index, ConstDictKeySource):
                raise ValueError(
                    "GetItemSource index must be a constant, enum or ConstDictKeySource"
                )
            return f"{self.base.name()}[{self.index.name()}]"
        elif self.index_is_slice:
            return f"{self.base.name()}[{self.unpack_slice()!r}]"
        elif isinstance(self.index, enum.Enum):
            return f"{self.base.name()}[{enum_repr(self.index, self.guard_source().is_local())}]"
        else:
            return f"{self.base.name()}[{self.index!r}]"

class LocalSource(Source):
    local_name: str
    cell_or_freevar: bool = False

    def reconstruct(self, codegen):
        codegen.append_output(codegen.create_load(self.local_name))

    def guard_source(self):
        return GuardSource.LOCAL

    def name(self):
        return f"L[{repr(self.local_name)}]"

