def validate_outputs_names_are_unique(outputs: List[JsonField]) -> None:
    output_names = get_output_names(outputs=outputs)
    if len(output_names) != len(outputs):
        raise DuplicatedNameError(
            public_message="Found duplicated input outputs names",
            context="workflow_compilation | specification_validation",
        )