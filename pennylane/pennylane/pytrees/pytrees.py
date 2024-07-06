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

