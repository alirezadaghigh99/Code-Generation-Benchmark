def filter_model_descriptions(
    descriptions: List[ModelDescription],
    model_id: str,
) -> Optional[ModelDescription]:
    matching_models = [d for d in descriptions if d.model_id == model_id]
    if len(matching_models) > 0:
        return matching_models[0]
    return None