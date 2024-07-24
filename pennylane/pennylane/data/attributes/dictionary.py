class DatasetDict(  # pylint: disable=too-many-ancestors
    Generic[T],
    DatasetAttribute[HDF5Group, typing.Mapping[str, T], typing.Mapping[str, T]],
    typing.MutableMapping[str, T],
    MapperMixin,
):
    """Provides a dict-like collection for Dataset attribute types. Keys must
    be strings."""

    type_id = "dict"

    def __post_init__(self, value: typing.Mapping[str, T]):
        super().__post_init__(value)
        self.update(value)

    @classmethod
    def default_value(cls) -> Dict:
        return {}

    def hdf5_to_value(self, bind: HDF5Group) -> typing.MutableMapping[str, T]:
        return self

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: None) -> HDF5Group:
        grp = bind_parent.create_group(key)

        return grp

    def copy_value(self) -> Dict[str, T]:
        return {key: attr.copy_value() for key, attr in self._mapper.items()}

    def copy(self) -> Dict[str, T]:
        """Returns a copy of this mapping as a builtin ``dict``, with all
        elements copied."""
        return self.copy_value()

    def __getitem__(self, __key: str) -> T:
        self._check_key(__key)

        return self._mapper[__key].get_value()

    def __setitem__(self, __key: str, __value: Union[T, DatasetAttribute[HDF5Any, T, T]]) -> None:
        self._check_key(__key)

        if __key in self:
            del self[__key]

        self._mapper[__key] = __value

    def __delitem__(self, __key: str) -> None:
        self._check_key(__key)

        del self._mapper[__key]

    def __len__(self) -> int:
        return len(self.bind)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Mapping):
            return False

        if not len(self) == len(__value):
            return False

        if self.keys() != __value.keys():
            return False

        return all(__value[key] == self[key] for key in __value.keys())

    def __iter__(self) -> typing.Iterator[str]:
        return (key for key in self.bind.keys())

    def __str__(self) -> str:
        return str(dict(self))

    def __repr__(self) -> str:
        return repr(dict(self))

    def _check_key(self, __key: str) -> None:
        """Checks that __key is a string, and raises a ``TypeError`` if it isn't."""
        if not isinstance(__key, str):
            raise TypeError(f"'{type(self).__name__}' keys must be strings.")

