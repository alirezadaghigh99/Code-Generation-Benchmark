def get_model_id_chunks(model_id: str) -> Tuple[DatasetID, VersionID]:
    model_id_chunks = model_id.split("/")
    if len(model_id_chunks) != 2:
        raise InvalidModelIDError(f"Model ID: `{model_id}` is invalid.")
    return model_id_chunks[0], model_id_chunks[1]