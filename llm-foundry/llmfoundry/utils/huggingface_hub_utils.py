def _flatten_import(
    node: ast.ImportFrom,
    flatten_imports_prefix: Sequence[str],
) -> bool:
    """Returns True if import should be flattened.

    Checks whether the node starts the same as any of the imports in
    flatten_imports_prefix.
    """
    for import_prefix in flatten_imports_prefix:
        if node.module is not None and node.module.startswith(import_prefix):
            return True
    return False