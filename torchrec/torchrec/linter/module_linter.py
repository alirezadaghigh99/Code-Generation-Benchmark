def linter_one_file(python_path: str) -> None:
    """
    This function will check all Modules defined in the given file for a valid
    documentation based on the AST.

    Input args:
        python_path: Path to the file that need to be verified with the linter.

    Returns:
        None
    """
    python_path = python_path.strip()
    try:
        for node in ast.parse(read_file(python_path)).body:
            if type(node) == ast.ClassDef:
                assert isinstance(node, ast.ClassDef)
                check_class_definition(python_path, node)
    except SyntaxError as e:  # pragma: nocover
        # possible failing due to file parsing error
        lint_item = {
            "path": python_path,
            "line": e.lineno,
            "char": e.offset,
            "severity": "warning",
            "name": "syntax-error",
            "description": (
                f"There is a linter parser error with message: {e.msg}. "
                "Please report the diff to torchrec oncall"
            ),
            "bypassChangedLineFiltering": True,
        }
        print(json.dumps(lint_item))