def is_installed(package: str) -> bool:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    # When executing `import mmengine.runner`,
    # pkg_resources will be imported and it takes too much time.
    # Therefore, import it in function scope to save time.
    import importlib.util

    import pkg_resources
    from pkg_resources import get_distribution

    # refresh the pkg_resources
    # more datails at https://github.com/pypa/setuptools/issues/373
    importlib.reload(pkg_resources)
    try:
        get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        spec = importlib.util.find_spec(package)
        if spec is None:
            return False
        elif spec.origin is not None:
            return True
        else:
            return False

