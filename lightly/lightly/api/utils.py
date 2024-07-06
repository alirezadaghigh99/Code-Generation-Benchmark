def getenv(key: str, default: str):
    """Return the value of the environment variable key if it exists,
    or default if it doesn’t.

    """
    try:
        return os.getenvb(key.encode(), default.encode()).decode()
    except Exception:
        pass
    try:
        return os.getenv(key, default)
    except Exception:
        pass
    return default

