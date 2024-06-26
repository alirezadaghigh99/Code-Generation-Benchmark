def get_parent_id_of_detections_from_sources(
    detections_from_sources: List[List[dict]],
) -> str:
    encountered_parent_ids = {
        p[PARENT_ID_KEY]
        for prediction_source in detections_from_sources
        for p in prediction_source
    }
    if len(encountered_parent_ids) != 1:
        raise ValueError(
            "Missmatch in predictions - while executing consensus step, "
            "in equivalent batches, detections are assigned different parent "
            "identifiers, whereas consensus can only be applied for predictions "
            "made against the same input."
        )
    return next(iter(encountered_parent_ids))