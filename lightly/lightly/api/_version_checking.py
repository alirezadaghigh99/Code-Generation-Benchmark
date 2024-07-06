def get_latest_version(
    current_version: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC
) -> str:
    """Returns the latest package version."""
    versioning_api = _get_versioning_api()
    version_number: str = versioning_api.get_latest_pip_version(
        current_version=current_version,
        _request_timeout=timeout_sec,
    )
    return version_number

def get_minimum_compatible_version(
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> str:
    """Returns minimum package version that is compatible with the API."""
    versioning_api = _get_versioning_api()
    version_number: str = versioning_api.get_minimum_compatible_pip_version(
        _request_timeout=timeout_sec
    )
    return version_number

