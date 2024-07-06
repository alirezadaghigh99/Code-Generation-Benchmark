def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())

