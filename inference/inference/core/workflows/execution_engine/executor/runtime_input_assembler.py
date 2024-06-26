def assembly_runtime_parameters(
    runtime_parameters: Dict[str, Any],
    defined_inputs: List[InputType],
) -> Dict[str, Any]:
    for defined_input in defined_inputs:
        if isinstance(defined_input, WorkflowImage):
            runtime_parameters[defined_input.name] = assembly_input_image(
                parameter=defined_input.name,
                image=runtime_parameters.get(defined_input.name),
            )
        else:
            runtime_parameters[defined_input.name] = assembly_inference_parameter(
                parameter=defined_input.name,
                runtime_parameters=runtime_parameters,
                default_value=defined_input.default_value,
            )
    return runtime_parameters