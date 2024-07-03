def get_output_names(outputs: List[JsonField]) -> Set[str]:
    return {construct_output_name(name=output.name) for output in outputs}