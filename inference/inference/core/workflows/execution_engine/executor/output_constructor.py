def convert_sv_detections_coordinates(data: Any) -> Any:
    if isinstance(data, sv.Detections):
        return sv_detections_to_root_coordinates(detections=data)
    if isinstance(data, dict):
        return {k: convert_sv_detections_coordinates(data=v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_sv_detections_coordinates(data=element) for element in data]
    return data

def data_contains_sv_detections(data: Any) -> bool:
    if isinstance(data, sv.Detections):
        return True
    if isinstance(data, dict):
        result = set()
        for value in data.values():
            result.add(data_contains_sv_detections(data=value))
        return True in result
    if isinstance(data, list):
        result = set()
        for value in data:
            result.add(data_contains_sv_detections(data=value))
        return True in result
    return False

def place_data_in_array(array: list, index: DynamicBatchIndex, data: Any) -> None:
    if len(index) == 0:
        raise ExecutionEngineRuntimeError(
            public_message=f"Reached end of index without possibility to place data in result array."
            f"This is most likely a bug. Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | output_construction",
        )
    elif len(index) == 1:
        array[index[0]] = data
    else:
        first_chunk, *remaining_index = index
        place_data_in_array(array=array[first_chunk], index=remaining_index, data=data)

def create_array(indices: np.ndarray) -> Optional[list]:
    if indices.size == 0:
        return None
    result = []
    max_idx = indices[:, 0].max() + 1
    for idx in range(max_idx):
        idx_selector = indices[:, 0] == idx
        indices_subset = indices[idx_selector][:, 1:]
        inner_array = create_array(indices_subset)
        if (
            inner_array is None
            and sum(indices_subset.shape) > 0
            and indices_subset.shape[0] == 0
        ):
            inner_array = create_empty_index_array(
                level=indices.shape[-1] - 1,
                accumulator=[],
            )
        result.append(inner_array)
    return result

