def torch_version_ge(major: int, minor: int, patch: Optional[int] = None) -> bool:
    _version = version.parse(torch_version())
    if patch is None:
        return _version >= version.parse(f"{major}.{minor}")
    else:
        return _version >= version.parse(f"{major}.{minor}.{patch}")

