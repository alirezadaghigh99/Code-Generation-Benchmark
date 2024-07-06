def check_optional_dependency(dependency_name):
    try:
        if dependency_name not in OPTIONAL_DEPENDENCIES.keys():
            raise ValueError(f'The given dependency \'{dependency_name}\' is not registered '
                             f'as an optional dependency.')

        importlib.import_module(OPTIONAL_DEPENDENCIES[dependency_name])
    except ImportError:
        exception_msg = f'The optional dependency \'{dependency_name}\' is required ' \
                        f'to use this functionality.'
        raise MissingOptionalDependencyError(exception_msg)

