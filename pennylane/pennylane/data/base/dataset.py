def open(
        cls,
        filepath: Union[str, Path],
        mode: Literal["w", "w-", "a", "r", "copy"] = "r",
    ) -> "Dataset":
        """Open existing dataset or create a new one at ``filepath``.

        Args:
            filepath: Path to dataset file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), "a" (append existing,
                create if doesn't exist), "r" (read existing, must exist), and "copy",
                which loads the dataset into memory and detaches it from the underlying
                file. Default is "r".
        Returns:
            Dataset object from file
        """
        filepath = Path(filepath).expanduser()

        if mode == "copy":
            with h5py.File(filepath, "r") as f_to_copy:
                f = hdf5.create_group()
                hdf5.copy_all(f_to_copy, f)
        else:
            f = h5py.File(filepath, mode)

        return cls(f)

class Dataset(MapperMixin, _DatasetTransform):
    """
    Base class for Datasets.
    """

    __data_name__: ClassVar[str]
    __identifiers__: ClassVar[Tuple[str, ...]]

    fields: ClassVar[typing.Mapping[str, Field]]
    """
    A mapping of attribute names to their ``Attribute`` information. Note that
    this contains attributes declared on the class, not attributes added to
    an instance. Use ``attrs`` to view all attributes on an instance.
    """

    bind_: Optional[HDF5Group] = _init_arg(default=None, alias="bind", kw_only=False)
    data_name_: Optional[str] = _init_arg(default=None, alias="data_name")

    def __init__(
        self,
        bind: Optional[HDF5Group] = None,
        *,
        data_name: Optional[str] = None,
        identifiers: Optional[Tuple[str, ...]] = None,
        **attrs: Any,
    ):
        """
        Load a dataset from a HDF5 Group or initialize a new Dataset.

        Args:
            bind: The HDF5 group that contains this dataset. If None, a new
                group will be created in memory. Any attributes that already exist
                in ``bind`` will be loaded into this dataset.
            data_name: String describing the type of data this datasets contains, e.g
                'qchem' for quantum chemistry. Defaults to the data name defined by
                the class, this is 'generic' for base datasets.
            identifiers: Tuple of names of attributes of this dataset that will serve
                as its parameters
            **attrs: Attributes to add to this dataset.
        """
        if isinstance(bind, (h5py.Group, h5py.File)):
            self._bind = bind
        else:
            self._bind = hdf5.create_group()

        self._init_bind(data_name, identifiers)

        for name in self.fields:
            try:
                attr_value = attrs.pop(name)
                setattr(self, name, attr_value)
            except KeyError:
                pass

        for name, attr in attrs.items():
            setattr(self, name, attr)

    @classmethod
    def open(
        cls,
        filepath: Union[str, Path],
        mode: Literal["w", "w-", "a", "r", "copy"] = "r",
    ) -> "Dataset":
        """Open existing dataset or create a new one at ``filepath``.

        Args:
            filepath: Path to dataset file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), "a" (append existing,
                create if doesn't exist), "r" (read existing, must exist), and "copy",
                which loads the dataset into memory and detaches it from the underlying
                file. Default is "r".
        Returns:
            Dataset object from file
        """
        filepath = Path(filepath).expanduser()

        if mode == "copy":
            with h5py.File(filepath, "r") as f_to_copy:
                f = hdf5.create_group()
                hdf5.copy_all(f_to_copy, f)
        else:
            f = h5py.File(filepath, mode)

        return cls(f)

    def close(self) -> None:
        """Close the underlying dataset file. The dataset will
        become inaccessible."""
        self.bind.close()

    @property
    def data_name(self) -> str:
        """Returns the data name (category) of this dataset."""
        return self.info.get("data_name", self.__data_name__)

    @property
    def identifiers(self) -> typing.Mapping[str, str]:  # pylint: disable=function-redefined
        """Returns this dataset's parameters."""
        return {
            attr_name: getattr(self, attr_name)
            for attr_name in self.info.get("identifiers", self.info.get("params", []))
            if attr_name in self.bind
        }

    @property
    def info(self) -> AttributeInfo:
        """Return metadata associated with this dataset."""
        return AttributeInfo(self.bind.attrs)

    @property
    def bind(self) -> HDF5Group:  # pylint: disable=function-redefined
        """Return the HDF5 group that contains this dataset."""
        return self._bind

    @property
    def attrs(self) -> typing.Mapping[str, DatasetAttribute]:
        """Returns all attributes of this Dataset."""
        return self._mapper.view()

    @property
    def attr_info(self) -> typing.Mapping[str, AttributeInfo]:
        """Returns a mapping of the ``AttributeInfo`` for each of this dataset's attributes."""
        return MappingProxyType(
            {
                attr_name: AttributeInfo(self.bind[attr_name].attrs)
                for attr_name in self.list_attributes()
            }
        )

    def list_attributes(self) -> List[str]:
        """Returns a list of this dataset's attributes."""
        return list(self.attrs.keys())

    def read(
        self,
        source: Union[str, Path, "Dataset"],
        attributes: Optional[typing.Iterable[str]] = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """Load dataset from HDF5 file at filepath.

        Args:
            source: Dataset, or path to HDF5 file containing dataset, from which
                to read attributes
            attributes: Optional list of attributes to copy. If None, all attributes
                will be copied.
            overwrite: Whether to overwrite attributes that already exist in this
                dataset.
        """
        if not isinstance(source, Dataset):
            source = Path(source).expanduser()
            source = Dataset.open(source, mode="r")

        source.write(self, attributes=attributes, overwrite=overwrite)

        source.close()

    def write(
        self,
        dest: Union[str, Path, "Dataset"],
        mode: Literal["w", "w-", "a"] = "a",
        attributes: Optional[typing.Iterable[str]] = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """Write dataset to HDF5 file at filepath.

        Args:
            dest: HDF5 file, or path to HDF5 file containing dataset, to write
                attributes to
            mode: File handling mode, if ``source`` is a file system path. Possible
                values are "w-" (create, fail if file exists), "w" (create, overwrite existing),
                and "a" (append existing, create if doesn't exist). Default is "w-".
            attributes: Optional list of attributes to copy. If None, all attributes
                will be copied. Note that identifiers will always be copied.
            overwrite: Whether to overwrite attributes that already exist in this
                dataset.
        """
        attributes = attributes if attributes is not None else ()
        on_conflict = "overwrite" if overwrite else "ignore"

        if not isinstance(dest, Dataset):
            dest = Path(dest).expanduser()
            dest = Dataset.open(dest, mode=mode)
            dest.info.update(self.info)

        hdf5.copy_all(self.bind, dest.bind, *attributes, on_conflict=on_conflict)

        missing_identifiers = [
            identifier for identifier in self.identifiers if not hasattr(dest, identifier)
        ]
        if missing_identifiers:
            hdf5.copy_all(self.bind, dest.bind, *missing_identifiers)

    def _init_bind(
        self, data_name: Optional[str] = None, identifiers: Optional[Tuple[str, ...]] = None
    ):
        if self.bind.file.mode == "r+":
            if "type_id" not in self.info:
                self.info["type_id"] = self.type_id
            if "data_name" not in self.info:
                self.info["data_name"] = data_name or self.__data_name__
            if "identifiers" not in self.info:
                self.info["identifiers"] = identifiers or self.__identifiers__

    def __setattr__(self, __name: str, __value: Union[Any, DatasetAttribute]) -> None:
        if __name.startswith("_") or __name in type(self).__dict__:
            object.__setattr__(self, __name, __value)
            return

        if __name in self.fields:
            field_ = self.fields[__name]
            self._mapper.set_item(__name, __value, field_.info, field_.attribute_type)
        else:
            self._mapper[__name] = __value

    def __getattr__(self, __name: str) -> Any:
        try:
            return self._mapper[__name].get_value()
        except KeyError as exc:
            if __name in self.fields:
                return UNSET

            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{__name}'"
            ) from exc

    def __delattr__(self, __name: str) -> None:
        try:
            del self._mapper[__name]
        except KeyError as exc:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{__name}'"
            ) from exc

    def __repr__(self) -> str:
        attrs_str = [repr(attr) for attr in self.list_attributes()]
        if len(attrs_str) > 2:
            attrs_str = attrs_str[:2]
            attrs_str.append("...")

        attrs_str = "[" + ", ".join(attrs_str) + "]"
        repr_items = ", ".join(
            f"{name}: {value}"
            for name, value in {**self.identifiers, "attributes": attrs_str}.items()
        )

        return f"<{type(self).__name__} = {repr_items}>"

    def __init_subclass__(
        cls, *, data_name: Optional[str] = None, identifiers: Optional[Tuple[str, ...]] = None
    ) -> None:
        """Initializes the ``fields`` dict of a Dataset subclass using
        the declared ``Attributes`` and their type annotations."""

        super().__init_subclass__()

        fields = {}
        if data_name:
            cls.__data_name__ = data_name
        if identifiers:
            cls.__identifiers__ = identifiers

        # get field info from annotated class attributes, e.g:
        # name: int = field(...)
        for name, annotated_type in cls.__annotations__.items():
            if get_origin(annotated_type) is ClassVar:
                continue

            try:
                field_ = getattr(cls, name)
                delattr(cls, name)
            except AttributeError:
                # field only has type annotation
                field_ = field()

            if field_ is _InitArg:
                continue

            field_.info.py_type = annotated_type
            if field_.attribute_type is UNSET:
                field_.attribute_type = match_obj_type(annotated_type)

            fields[name] = field_

        cls.fields = MappingProxyType(fields)

    def __dir__(self):
        return self.list_attributes()

    __data_name__ = "generic"
    __identifiers__ = tuple()

    type_id = "dataset"
    """Type identifier for this dataset. Used internally to load datasets
    from other datasets."""

