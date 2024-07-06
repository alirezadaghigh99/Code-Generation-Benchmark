def retrieve_init_parameter_values(
    block_name: str,
    block_init_parameter: str,
    block_source: str,
    explicit_init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
    initializers: Dict[str, Union[Any, Callable[[None], Any]]],
) -> Any:
    full_parameter_name = f"{block_source}.{block_init_parameter}"
    if full_parameter_name in explicit_init_parameters:
        return explicit_init_parameters[full_parameter_name]
    if full_parameter_name in initializers:
        return call_if_callable(initializers[full_parameter_name])
    if block_init_parameter in explicit_init_parameters:
        return explicit_init_parameters[block_init_parameter]
    if block_init_parameter in initializers:
        return call_if_callable(initializers[block_init_parameter])
    raise BlockInitParameterNotProvidedError(
        public_message=f"Could not resolve init parameter {block_init_parameter} to initialise "
        f"step `{block_name}` from plugin: {block_source}.",
        context="workflow_compilation | steps_initialisation",
    )

