def get_model_type(
    model_id: str,
    api_key: Optional[str] = None,
) -> Tuple[TaskType, ModelType]:
    """Retrieves the model type based on the given model ID and API key.

    Args:
        model_id (str): The ID of the model.
        api_key (str): The API key used to authenticate.

    Returns:
        tuple: The project task type and the model type.

    Raises:
        WorkspaceLoadError: If the workspace could not be loaded or if the API key is invalid.
        DatasetLoadError: If the dataset could not be loaded due to invalid ID, workspace ID or version ID.
        MissingDefaultModelError: If default model is not configured and API does not provide this info
        MalformedRoboflowAPIResponseError: Roboflow API responds in invalid format.
    """
    model_id = resolve_roboflow_model_alias(model_id=model_id)
    dataset_id, version_id = get_model_id_chunks(model_id=model_id)
    if dataset_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {dataset_id}.")
        return GENERIC_MODELS[dataset_id]
    cached_metadata = get_model_metadata_from_cache(
        dataset_id=dataset_id, version_id=version_id
    )
    if cached_metadata is not None:
        return cached_metadata[0], cached_metadata[1]
    if version_id == STUB_VERSION_ID:
        if api_key is None:
            raise MissingApiKeyError(
                "Stub model version provided but no API key was provided. API key is required to load stub models."
            )
        workspace_id = get_roboflow_workspace(api_key=api_key)
        project_task_type = get_roboflow_dataset_type(
            api_key=api_key, workspace_id=workspace_id, dataset_id=dataset_id
        )
        model_type = "stub"
        save_model_metadata_in_cache(
            dataset_id=dataset_id,
            version_id=version_id,
            project_task_type=project_task_type,
            model_type=model_type,
        )
        return project_task_type, model_type
    api_data = get_roboflow_model_data(
        api_key=api_key,
        model_id=model_id,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    ).get("ort")
    if api_data is None:
        raise ModelArtefactError("Error loading model artifacts from Roboflow API.")
    # some older projects do not have type field - hence defaulting
    project_task_type = api_data.get("type", "object-detection")
    model_type = api_data.get("modelType")
    if model_type is None or model_type == "ort":
        # some very old model versions do not have modelType reported - and API respond in a generic way -
        # then we shall attempt using default model for given task type
        model_type = MODEL_TYPE_DEFAULTS.get(project_task_type)
    if model_type is None or project_task_type is None:
        raise ModelArtefactError("Error loading model artifacts from Roboflow API.")
    save_model_metadata_in_cache(
        dataset_id=dataset_id,
        version_id=version_id,
        project_task_type=project_task_type,
        model_type=model_type,
    )

    return project_task_type, model_type

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

