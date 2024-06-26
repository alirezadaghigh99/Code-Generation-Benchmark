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