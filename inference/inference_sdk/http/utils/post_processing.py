def decode_workflow_outputs(
    workflow_outputs: Dict[str, Any],
    expected_format: VisualisationResponseFormat,
) -> Dict[str, Any]:
    result = {}
    for key, value in workflow_outputs.items():
        if is_workflow_image(value=value):
            value = decode_workflow_output_image(
                value=value,
                expected_format=expected_format,
            )
        elif issubclass(type(value), dict):
            value = decode_workflow_outputs(
                workflow_outputs=value, expected_format=expected_format
            )
        elif issubclass(type(value), list):
            value = decode_workflow_output_list(
                elements=value,
                expected_format=expected_format,
            )
        result[key] = value
    return result