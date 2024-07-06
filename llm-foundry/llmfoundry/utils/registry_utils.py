def create_registry(
    *namespace: str,
    generic_type: Type[S],
    entry_points: bool = False,
    description: str = '',
) -> 'TypedRegistry[S]':
    """Create a new registry.

    Args:
        namespace (str): The namespace, e.g. "llmfoundry.loggers"
        generic_type (Type[S]): The type of the registry.
        entry_points (bool): Accept registered functions from entry points.
        description (str): A description of the registry.

    Returns:
        The TypedRegistry object.
    """
    if catalogue.check_exists(*namespace):
        raise catalogue.RegistryError(f'Namespace already exists: {namespace}')

    return TypedRegistry[generic_type](
        namespace,
        entry_points=entry_points,
        description=description,
    )

def import_file(loc: Union[str, Path]) -> ModuleType:
    """Import module from a file.

    Used to run arbitrary python code.
    Args:
        name (str): Name of module to load.
        loc (str / Path): Path to the file.

    Returns:
        ModuleType: The module object.
    """
    if not os.path.exists(loc):
        raise FileNotFoundError(f'File {loc} does not exist.')

    spec = importlib.util.spec_from_file_location('python_code', str(loc))

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f'Error executing {loc}') from e
    return module

