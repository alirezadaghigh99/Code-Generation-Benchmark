def tree_leaves(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> List[Any]:
    """Get a list of leaves of a pytree."""
    return list(tree_iter(tree, is_leaf=is_leaf))

def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': True}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    """
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, *flat_args))

def tree_flatten(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Tuple[List[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """
    leaves: List[Any] = []
    spec = _tree_flatten_helper(tree, leaves, is_leaf=is_leaf)
    return leaves, spec

def tree_map_with_path(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Like :func:`tree_map`, but the provided callable takes an additional key path argument.

    Args:
        func: A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees. The first positional argument
            to ``func`` is the key path of the leaf in question. The second
            positional argument is the value of the leaf.
        tree: A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests: A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(keypath, x, *xs)`` where ``keypath`` is the key path at the
        corresponding leaf in ``tree``, ``x`` is the value at that leaf, and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    keypath_leaves, treespec = tree_flatten_with_path(tree, is_leaf)
    keypath_leaves = list(zip(*keypath_leaves))
    all_keypath_leaves = keypath_leaves + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(func(*xs) for xs in zip(*all_keypath_leaves))

