def show_progress(
    log_line: dict, progress: Progress, progress_tasks: Dict[str, TaskID]
) -> None:
    log_id, status = log_line.get("id"), log_line.get("status")
    if log_line["status"].lower() == "downloading":
        task_id = f"[red][Downloading {log_id}]"
    elif log_line["status"].lower() == "extracting":
        task_id = f"[green][Extracting {log_id}]"
    else:
        return None
    if task_id not in progress_tasks:
        progress_tasks[task_id] = progress.add_task(
            f"{task_id}", total=log_line.get("progressDetail", {}).get("total")
        )
    else:
        progress.update(
            progress_tasks[task_id],
            completed=log_line.get("progressDetail", {}).get("current"),
        )

def is_container_running(container: Container) -> str:
    return container.attrs.get("State", {}).get("Status", "").lower() == "running"

def terminate_running_containers(
    containers: List[Container], interactive_mode: bool = True
) -> bool:
    """
    Args:
        containers (List[Container]): List of containers to handle
        interactive_mode (bool): Flag to determine if user prompt should decide on container termination

    Returns: boolean value that informs if there are containers that have not received SIGKILL
        as a result of procedure.
    """
    running_inference_containers = [
        c for c in containers if is_container_running(container=c)
    ]
    containers_to_kill = running_inference_containers
    if interactive_mode:
        containers_to_kill = [
            c for c in running_inference_containers if ask_user_to_kill_container(c)
        ]
    kill_containers(containers=containers_to_kill)
    return len(containers_to_kill) < len(running_inference_containers)

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

