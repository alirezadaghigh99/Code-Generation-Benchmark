def load_pandas_module():
    if not pkgutil.find_loader('pandas'):
        raise ImportError(
            "The `pandas` library is not installed. Try to "
            "install it with pip: \n    pip install pandas")

    return importlib.import_module('pandas')