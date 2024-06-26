def model_metadata_content_is_invalid(content: Optional[Union[list, dict]]) -> bool:
    if content is None:
        logger.warning("Empty model metadata file encountered in cache.")
        return True
    if not issubclass(type(content), dict):
        logger.warning("Malformed file encountered in cache.")
        return True
    if PROJECT_TASK_TYPE_KEY not in content or MODEL_TYPE_KEY not in content:
        logger.warning(
            f"Could not find one of required keys {PROJECT_TASK_TYPE_KEY} or {MODEL_TYPE_KEY} in cache."
        )
        return True
    return False