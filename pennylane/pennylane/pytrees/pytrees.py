def get_typename(pytree_type: type[Any]) -> str:
    """Return the typename under which ``pytree_type``
    was registered.

    Raises:
        TypeError: If ``pytree_type`` is not a Pytree.

    >>> get_typename(list)
    'builtins.list'
    >>> import pennylane
    >>> get_typename(pennylane.PauliX)
    'qml.PauliX'
    """

    try:
        return type_to_typename[pytree_type]
    except KeyError as exc:
        raise TypeError(f"{repr(pytree_type)} is not a Pytree type") from exc

def get_typename_type(typename: str) -> type[Any]:
    """Return the Pytree type with given ``typename``.

    Raises:
        ValueError: If ``typename`` is not the name of a
            pytree type.

    >>> import pennylane
    >>> get_typename_type("builtins.list")
    <class 'list'>
    >>> get_typename_type("qml.PauliX")
    <class 'pennylane.ops.qubit.non_parametric_ops.PauliX'>
    """
    try:
        return typename_to_type[typename]
    except KeyError as exc:
        raise ValueError(f"{repr(typename)} is not the name of a Pytree type.") from exc

class PyTreeStructure:
    """A pytree data structure, holding the type, metadata, and child pytree structures.

    >>> op = qml.adjoint(qml.RX(0.1, 0))
    >>> data, structure = qml.pytrees.flatten(op)
    >>> structure
    PyTree(AdjointOperation, (), [PyTree(RX, (<Wires = [0]>, ()), [Leaf])])

    A leaf is defined as just a ``PyTreeStructure`` with ``type_=None``.
    """

    type_: Optional[type[Any]] = None
    """The type corresponding to the node. If ``None``, then the structure is a leaf."""

    metadata: Metadata = ()
    """Any metadata needed to reproduce the original object."""

    children: list["PyTreeStructure"] = field(default_factory=list)
    """The children of the pytree node.  Can be either other structures or terminal leaves."""

    @property
    def is_leaf(self) -> bool:
        """Whether or not the structure is a leaf."""
        return self.type_ is None

    def __repr__(self):
        if self.is_leaf:
            return "PyTreeStructure()"
        return f"PyTreeStructure({self.type_.__name__}, {self.metadata}, {self.children})"

    def __str__(self):
        if self.is_leaf:
            return "Leaf"
        children_string = ", ".join(str(c) for c in self.children)
        return f"PyTree({self.type_.__name__}, {self.metadata}, [{children_string}])"

