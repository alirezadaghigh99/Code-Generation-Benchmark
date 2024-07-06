def register_image_at_roboflow(
    api_key: str,
    dataset_id: DatasetID,
    local_image_id: str,
    image_bytes: bytes,
    batch_name: str,
    tags: Optional[List[str]] = None,
    inference_id: Optional[str] = None,
) -> dict:
    url = f"{API_BASE_URL}/dataset/{dataset_id}/upload"
    params = [
        ("api_key", api_key),
        ("batch", batch_name),
    ]
    if inference_id is not None:
        params.append(("inference_id", inference_id))
    tags = tags if tags is not None else []
    for tag in tags:
        params.append(("tag", tag))
    wrapped_url = wrap_url(_add_params_to_url(url=url, params=params))
    m = MultipartEncoder(
        fields={
            "name": f"{local_image_id}.jpg",
            "file": ("imageToUpload", image_bytes, "image/jpeg"),
        }
    )
    response = requests.post(
        url=wrapped_url,
        data=m,
        headers={"Content-Type": m.content_type},
    )
    api_key_safe_raise_for_status(response=response)
    parsed_response = response.json()
    if not parsed_response.get("duplicate") and not parsed_response.get("success"):
        raise RoboflowAPIImageUploadRejectionError(
            f"Server rejected image: {parsed_response}"
        )
    return parsed_response

def annotate_image_at_roboflow(
    api_key: str,
    dataset_id: DatasetID,
    local_image_id: str,
    roboflow_image_id: str,
    annotation_content: str,
    annotation_file_type: str,
    is_prediction: bool = True,
) -> dict:
    url = f"{API_BASE_URL}/dataset/{dataset_id}/annotate/{roboflow_image_id}"
    params = [
        ("api_key", api_key),
        ("name", f"{local_image_id}.{annotation_file_type}"),
        ("prediction", str(is_prediction).lower()),
    ]
    wrapped_url = wrap_url(_add_params_to_url(url=url, params=params))
    response = requests.post(
        wrapped_url,
        data=annotation_content,
        headers={"Content-Type": "text/plain"},
    )
    api_key_safe_raise_for_status(response=response)
    parsed_response = response.json()
    if "error" in parsed_response or not parsed_response.get("success"):
        raise RoboflowAPIIAnnotationRejectionError(
            f"Failed to save annotation for {roboflow_image_id}. API response: {parsed_response}"
        )
    return parsed_response

def get_roboflow_model_type(
    api_key: str,
    workspace_id: WorkspaceID,
    dataset_id: DatasetID,
    version_id: VersionID,
    project_task_type: ModelType,
) -> ModelType:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}/{version_id}",
        params=[("api_key", api_key), ("nocache", "true")],
    )
    version_info = _get_from_url(url=api_url)
    model_type = version_info["version"]
    if "modelType" not in model_type:
        if project_task_type not in MODEL_TYPE_DEFAULTS:
            raise MissingDefaultModelError(
                f"Could not set default model for {project_task_type}"
            )
        logger.warning(
            f"Model type not defined - using default for {project_task_type} task."
        )
    return model_type.get("modelType", MODEL_TYPE_DEFAULTS[project_task_type])

def get_workflow_specification(
    api_key: str,
    workspace_id: WorkspaceID,
    workflow_id: str,
) -> dict:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/workflows/{workflow_id}",
        params=[("api_key", api_key)],
    )
    response = _get_from_url(url=api_url)
    if "workflow" not in response or "config" not in response["workflow"]:
        raise MalformedWorkflowResponseError(
            f"Could not find workflow specification in API response"
        )
    try:
        workflow_config = json.loads(response["workflow"]["config"])
        return workflow_config["specification"]
    except KeyError as error:
        raise MalformedWorkflowResponseError(
            "Workflow specification not found in Roboflow API response"
        ) from error
    except (ValueError, TypeError) as error:
        raise MalformedWorkflowResponseError(
            "Could not decode workflow specification in Roboflow API response"
        ) from error

def get_roboflow_labeling_batches(
    api_key: str, workspace_id: WorkspaceID, dataset_id: str
) -> dict:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}/batches",
        params=[("api_key", api_key)],
    )
    return _get_from_url(url=api_url)

def get_roboflow_labeling_jobs(
    api_key: str, workspace_id: WorkspaceID, dataset_id: str
) -> dict:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}/jobs",
        params=[("api_key", api_key)],
    )
    return _get_from_url(url=api_url)

def get_roboflow_dataset_type(
    api_key: str, workspace_id: WorkspaceID, dataset_id: DatasetID
) -> TaskType:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}",
        params=[("api_key", api_key), ("nocache", "true")],
    )
    dataset_info = _get_from_url(url=api_url)
    project_task_type = dataset_info.get("project", {})
    if "type" not in project_task_type:
        logger.warning(
            f"Project task type not defined for workspace={workspace_id} and dataset={dataset_id}, defaulting "
            f"to object-detection."
        )
    return project_task_type.get("type", "object-detection")

def get_roboflow_model_data(
    api_key: str,
    model_id: str,
    endpoint_type: ModelEndpointType,
    device_id: str,
) -> dict:
    api_data_cache_key = f"roboflow_api_data:{endpoint_type.value}:{model_id}"
    api_data = cache.get(api_data_cache_key)
    if api_data is not None:
        logger.debug(f"Loaded model data from cache with key: {api_data_cache_key}.")
        return api_data
    else:
        params = [
            ("nocache", "true"),
            ("device", device_id),
            ("dynamic", "true"),
        ]
        if api_key is not None:
            params.append(("api_key", api_key))
        api_url = _add_params_to_url(
            url=f"{API_BASE_URL}/{endpoint_type.value}/{model_id}",
            params=params,
        )
        api_data = _get_from_url(url=api_url)
        cache.set(
            api_data_cache_key,
            api_data,
            expire=10,
        )
        logger.debug(
            f"Loaded model data from Roboflow API and saved to cache with key: {api_data_cache_key}."
        )
        return api_data

def get_roboflow_workspace(api_key: str) -> WorkspaceID:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/",
        params=[("api_key", api_key), ("nocache", "true")],
    )
    api_key_info = _get_from_url(url=api_url)
    workspace_id = api_key_info.get("workspace")
    if workspace_id is None:
        raise WorkspaceLoadError(f"Empty workspace encountered, check your API key.")
    return workspace_id

def get_roboflow_active_learning_configuration(
    api_key: str,
    workspace_id: WorkspaceID,
    dataset_id: DatasetID,
) -> dict:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}/active_learning",
        params=[("api_key", api_key)],
    )
    return _get_from_url(url=api_url)

