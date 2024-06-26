def prepare_container_environment(
    port: int,
    project: str,
    metrics_enabled: bool,
    device_id: Optional[str],
    num_workers: int,
    api_key: Optional[str],
    env_file_path: Optional[str],
    development: bool = False,
) -> List[str]:
    environment = {}
    if env_file_path is not None:
        environment = read_env_file(path=env_file_path)
    environment["HOST"] = "0.0.0.0"
    environment["PORT"] = str(port)
    environment["PROJECT"] = project
    environment["METRICS_ENABLED"] = str(metrics_enabled)
    if device_id is not None:
        environment["DEVICE_ID"] = device_id
    if api_key is not None:
        environment["ROBOFLOW_API_KEY"] = api_key
    environment["NUM_WORKERS"] = str(num_workers)
    if development:
        environment["NOTEBOOK_ENABLED"] = "True"
    return [f"{key}={value}" for key, value in environment.items()]