def validate_input_mount(input_mount: Path) -> None:
    """Validates that the input mount is a directory and contains files."""
    input_mount = input_mount.resolve()
    if not input_mount.exists():
        raise ValueError(
            f"Path for 'input_mount' argument '{input_mount}' does not exist."
        )
    if not input_mount.is_dir():
        raise ValueError(
            f"Path for 'input_mount' argument '{input_mount}' is not a directory."
        )
    if not _dir_contains_image_or_video(path=input_mount):
        raise ValueError(
            f"Path for 'input_mount' argument '{input_mount}' does not contain any "
            "images or videos. Please verify that this is the correct directory. See "
            "our docs on lightly-serve for more information: "
            "https://docs.lightly.ai/docs/local-storage#optional-after-run-view-local-data-in-lightly-platform"
        )

def _translate_path(path: str, directories: Sequence[Path]) -> str:
    """Translates a relative path to a file in the local datasource.

    Tries to resolve the relative path to a file in the first directory
    and serves it if it exists. Otherwise, it tries to resolve the relative
    path to a file in the second directory and serves it if it exists, etc.

    Args:
        path:
            Relative path to a file in the local datasource.
        directories:
            List of directories to search for the file.


    Returns:
        Absolute path to the file in the local datasource or an empty string
        if the file doesn't exist.

    """
    path = parse.unquote(path)
    stripped_path = path.lstrip("/")
    for directory in directories:
        if (directory / stripped_path).exists():
            return str(directory / stripped_path)
    return ""  # Not found.

