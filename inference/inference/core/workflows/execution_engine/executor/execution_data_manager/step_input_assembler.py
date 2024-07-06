def ensure_compound_input_indices_match(indices: List[List[DynamicBatchIndex]]) -> None:
    if len(indices) < 2:
        return None
    reference_set = set(indices[0])
    for index in indices[1:]:
        other_set = set(index)
        if reference_set != other_set:
            raise ExecutionEngineRuntimeError(
                public_message=f"Detected a situation when step input parameters cannot be created "
                f"due to missmatch in batch element indices. This is most likely a bug. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )

def unfold_parameters(
    parameters: Dict[str, Any]
) -> Generator[Dict[str, Any], None, None]:
    batch_parameters = get_batch_parameters(parameters=parameters)
    non_batch_parameters = {
        k: v for k, v in parameters.items() if k not in batch_parameters
    }
    if not batch_parameters:
        if not non_batch_parameters:
            return None
        yield non_batch_parameters
        return None
    for unfolded_batch_parameters in iterate_over_batches(
        batch_parameters=batch_parameters
    ):
        yield {**unfolded_batch_parameters, **non_batch_parameters}

def get_empty_batch_elements_indices(value: Any) -> Set[DynamicBatchIndex]:
    result = set()
    if isinstance(value, dict):
        for v in value.values():
            value_result = get_empty_batch_elements_indices(v)
            result = result.union(value_result)
    if isinstance(value, list):
        for v in value:
            value_result = get_empty_batch_elements_indices(v)
            result = result.union(value_result)
    if isinstance(value, Batch):
        for index, value_element in value.iter_with_indices():
            if isinstance(value_element, Batch):
                value_result = get_empty_batch_elements_indices(value=value_element)
                result = result.union(value_result)
            elif value_element is None:
                result.add(index)
    return result

