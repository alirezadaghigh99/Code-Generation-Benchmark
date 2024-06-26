def retrieve_init_parameters_values(
    block_init_parameters: List[str],
    block_source: str,
    explicit_init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
    initializers: Dict[str, Union[Any, Callable[[None], Any]]],
) -> Dict[str, Any]:
    return {
        block_init_parameter: retrieve_init_parameter_values(
            block_init_parameter=block_init_parameter,
            block_source=block_source,
            explicit_init_parameters=explicit_init_parameters,
            initializers=initializers,
        )
        for block_init_parameter in block_init_parameters
    }