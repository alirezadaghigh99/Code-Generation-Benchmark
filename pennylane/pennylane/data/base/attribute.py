class AttributeInfo(MutableMapping):
    """Contains metadata that may be assigned to a dataset
    attribute. Is stored in the HDF5 object's ``attrs`` dict.

    Attributes:
        attrs_bind: The HDF5 attrs dict that this instance is bound to,
            or any mutable mapping
        **kwargs: Extra metadata to include. Must be a string, number
            or numpy array
    """

    attrs_namespace: ClassVar[str] = "qml.data"
    attrs_bind: typing.MutableMapping[str, Any]

    @overload
    def __init__(  # overload to specify known keyword args
        self,
        attrs_bind: Optional[typing.MutableMapping[str, Any]] = None,
        *,
        doc: Optional[str] = None,
        py_type: Optional[str] = None,
        **kwargs: Any,
    ):
        pass

    @overload
    def __init__(self):  # need at least two overloads when using @overload
        pass

    def __init__(self, attrs_bind: Optional[typing.MutableMapping[str, Any]] = None, **kwargs: Any):
        object.__setattr__(self, "attrs_bind", attrs_bind if attrs_bind is not None else {})

        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, info: "AttributeInfo") -> None:
        """Inserts the values set in this instance into ``info``."""
        for k, v in self.items():
            info[k] = v

    def load(self, info: "AttributeInfo"):
        """Inserts the values set in ``info`` into this instance."""
        info.save(self)

    @property
    def py_type(self) -> Optional[str]:
        """String representation of this attribute's python type."""
        return self.get("py_type")

    @py_type.setter
    def py_type(self, type_: Union[str, Type]):
        self["py_type"] = get_type_str(type_)

    @property
    def doc(self) -> Optional[str]:
        """Documentation for this attribute."""
        return self.get("doc")

    @doc.setter
    def doc(self, doc: str):
        self["doc"] = doc

    def __len__(self) -> int:
        return self.attrs_bind.get("qml.__data_len__", 0)

    def _update_len(self, inc: int):
        self.attrs_bind["qml.__data_len__"] = len(self) + inc

    def __setitem__(self, __name: str, __value: Any):
        key = self.bind_key(__name)
        if __value is None:
            self.attrs_bind.pop(key, None)
            return

        exists = key in self.attrs_bind
        self.attrs_bind[key] = __value
        if not exists:
            self._update_len(1)

    def __getitem__(self, __name: str) -> Any:
        return self.attrs_bind[self.bind_key(__name)]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.__class__.__dict__:
            object.__setattr__(self, __name, __value)
        else:
            self[__name] = __value

    def __getattr__(self, __name: str) -> Any:
        try:
            return self[__name]
        except KeyError:
            return None

    def __delitem__(self, __name: str) -> None:
        del self.attrs_bind[self.bind_key(__name)]
        self._update_len(-1)

    def __iter__(self) -> Iterator[str]:
        ns = f"{self.attrs_namespace}."

        return (
            key.split(ns, maxsplit=1)[1]
            for key in filter(lambda k: k.startswith(ns), self.attrs_bind)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(dict(self))})"

    @classmethod
    @lru_cache()
    def bind_key(cls, __name: str) -> str:
        """Returns ``__name`` dot-prefixed with ``attrs_namespace``."""
        return ".".join((cls.attrs_namespace, __name))