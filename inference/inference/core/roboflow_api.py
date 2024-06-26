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