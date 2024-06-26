def kill_containers(containers: List[Container]) -> None:
    for container in containers:
        container.kill()