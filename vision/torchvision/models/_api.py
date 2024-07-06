def get_model_weights(name: Union[Callable, str]) -> Type[WeightsEnum]:
    """
    Returns the weights enum class associated to the given model.

    Args:
        name (callable or str): The model builder function or the name under which it is registered.

    Returns:
        weights_enum (WeightsEnum): The weights enum class associated with the model.
    """
    model = get_model_builder(name) if isinstance(name, str) else name
    return _get_enum_from_fn(model)

def list_models(
    module: Optional[ModuleType] = None,
    include: Union[Iterable[str], str, None] = None,
    exclude: Union[Iterable[str], str, None] = None,
) -> List[str]:
    """
    Returns a list with the names of registered models.

    Args:
        module (ModuleType, optional): The module from which we want to extract the available models.
        include (str or Iterable[str], optional): Filter(s) for including the models from the set of all models.
            Filters are passed to `fnmatch <https://docs.python.org/3/library/fnmatch.html>`__ to match Unix shell-style
            wildcards. In case of many filters, the results is the union of individual filters.
        exclude (str or Iterable[str], optional): Filter(s) applied after include_filters to remove models.
            Filter are passed to `fnmatch <https://docs.python.org/3/library/fnmatch.html>`__ to match Unix shell-style
            wildcards. In case of many filters, the results is removal of all the models that match any individual filter.

    Returns:
        models (list): A list with the names of available models.
    """
    all_models = {
        k for k, v in BUILTIN_MODELS.items() if module is None or v.__module__.rsplit(".", 1)[0] == module.__name__
    }
    if include:
        models: Set[str] = set()
        if isinstance(include, str):
            include = [include]
        for include_filter in include:
            models = models | set(fnmatch.filter(all_models, include_filter))
    else:
        models = all_models

    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]
        for exclude_filter in exclude:
            models = models - set(fnmatch.filter(all_models, exclude_filter))
    return sorted(models)

def get_model(name: str, **config: Any) -> nn.Module:
    """
    Gets the model name and configuration and returns an instantiated model.

    Args:
        name (str): The name under which the model is registered.
        **config (Any): parameters passed to the model builder method.

    Returns:
        model (nn.Module): The initialized model.
    """
    fn = get_model_builder(name)
    return fn(**config)

def get_model_builder(name: str) -> Callable[..., nn.Module]:
    """
    Gets the model name and returns the model builder method.

    Args:
        name (str): The name under which the model is registered.

    Returns:
        fn (Callable): The model builder method.
    """
    name = name.lower()
    try:
        fn = BUILTIN_MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name}")
    return fn

