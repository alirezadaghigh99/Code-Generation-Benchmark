def get_filesystem(path: str, **kwargs: Any) -> fsspec.AbstractFileSystem:
    """Returns the appropriate filesystem to use when handling the given path."""
    fs, _ = url_to_fs(path, **kwargs)
    return fs

