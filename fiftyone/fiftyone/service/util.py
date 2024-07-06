def get_listening_tcp_ports(process):
    """Retrieves a list of TCP ports that the specified process is listening on.

    Args:
        process (psutil.Process): the process to check

    Returns:
        generator of integers
    """
    for conn in process.connections(kind="tcp"):
        if (
            not conn.raddr  # not connected to a remote socket
            and conn.status == psutil.CONN_LISTEN
        ):
            yield conn.laddr[1]  # port

def find_processes_by_args(args):
    """Finds a process with the specified command-line arguments.

    Only processes for the current user will be returned.

    Args:
        args (list[str]): a list of arguments, in the order to search for

    Returns:
        generator of psutil.Process objects
    """
    if not isinstance(args, list):
        raise TypeError("args must be list")
    if not args:
        raise ValueError("empty search")

    current_username = psutil.Process().username()
    for p in psutil.process_iter(["cmdline", "username"]):
        try:
            if p.info["username"] == current_username and p.info["cmdline"]:
                cmdline = p.info["cmdline"]
                for i in range(len(cmdline) - len(args) + 1):
                    if cmdline[i : i + len(args)] == args:
                        if not _is_wrapper_process(p):
                            yield p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

