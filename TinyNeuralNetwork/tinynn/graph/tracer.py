def fetch_modules(config: typing.Optional[str] = None):
    """Fetches the functions from the config."""
    if config is None:
        config = os.path.join(current_dir, 'configs/torch_module_override.yml')
    modules = []
    with open(config, 'r') as f:
        module_dict = yaml.load(f, yaml.SafeLoader)
        for ns, module_names in module_dict.items():
            try:
                scope = importlib.import_module(ns)
            except ImportError:
                pass
            for module_name in module_names:
                if hasattr(scope, module_name):
                    module = getattr(scope, module_name)
                    modules.append(module)
                    importable_module_names[module] = f'{ns}.{module_name}'
                    if hasattr(module, '__init__'):
                        constructor = module.__init__
                        module_constructor_signatures[module] = inspect.signature(constructor).parameters.values()
    return modules