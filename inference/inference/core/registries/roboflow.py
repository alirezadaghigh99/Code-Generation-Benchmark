def get_model_metadata_from_cache(
    dataset_id: str, version_id: str
) -> Optional[Tuple[TaskType, ModelType]]:
    if LAMBDA:
        return _get_model_metadata_from_cache(
            dataset_id=dataset_id, version_id=version_id
        )
    with cache.lock(
        f"lock:metadata:{dataset_id}:{version_id}", expire=CACHE_METADATA_LOCK_TIMEOUT
    ):
        return _get_model_metadata_from_cache(
            dataset_id=dataset_id, version_id=version_id
        )