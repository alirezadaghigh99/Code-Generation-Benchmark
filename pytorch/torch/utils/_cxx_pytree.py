def tree_flatten(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Tuple[List[Any], TreeSpec]:
    """Flatten a pytree.

    See also :func:`tree_unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(tree)
    ([1, 2, 3, 4, None, 5], PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf))
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*, NoneIsLeaf))
    >>> tree_flatten(None)
    ([None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten(tree)
    ([2, 3, 4, 1, None, 5], PyTreeSpec(OrderedDict([('b', (*, [*, *])), ('a', *), ('c', *), ('d', *)]), NoneIsLeaf))

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    """
    return optree.tree_flatten(  # type: ignore[return-value]
        tree,
        is_leaf=is_leaf,
        none_is_leaf=True,
        namespace="torch",
    )

