def _matches_machine_hostname(host: str) -> bool:
    """Indicate whether ``host`` matches the hostname of this machine.

    This function compares ``host`` to the hostname as well as to the IP
    addresses of this machine. Note that it may return a false negative if this
    machine has CNAME records beyond its FQDN or IP addresses assigned to
    secondary NICs.
    """
    if host == "localhost":
        return True

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None

    if addr and addr.is_loopback:
        return True

    try:
        host_addr_list = socket.getaddrinfo(
            host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
        )
    except (ValueError, socket.gaierror) as _:
        host_addr_list = []

    host_ip_list = [host_addr_info[4][0] for host_addr_info in host_addr_list]

    this_host = socket.gethostname()
    if host == this_host:
        return True

    addr_list = socket.getaddrinfo(
        this_host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
    )
    for addr_info in addr_list:
        # If we have an FQDN in the addr_info, compare it to `host`.
        if addr_info[3] and addr_info[3] == host:
            return True

        # Otherwise if `host` represents an IP address, compare it to our IP
        # address.
        if addr and addr_info[4][0] == str(addr):
            return True

        # If the IP address matches one of the provided host's IP addresses
        if addr_info[4][0] in host_ip_list:
            return True

    return Falsedef parse_rendezvous_endpoint(
    endpoint: Optional[str], default_port: int
) -> Tuple[str, int]:
    """Extract the hostname and the port number from a rendezvous endpoint.

    Args:
        endpoint:
            A string in format <hostname>[:<port>].
        default_port:
            The port number to use if the endpoint does not include one.

    Returns:
        A tuple of hostname and port number.
    """
    if endpoint is not None:
        endpoint = endpoint.strip()

    if not endpoint:
        return ("localhost", default_port)

    # An endpoint that starts and ends with brackets represents an IPv6 address.
    if endpoint[0] == "[" and endpoint[-1] == "]":
        host, *rest = endpoint, *[]
    else:
        host, *rest = endpoint.rsplit(":", 1)

    # Sanitize the IPv6 address.
    if len(host) > 1 and host[0] == "[" and host[-1] == "]":
        host = host[1:-1]

    if len(rest) == 1:
        port = _try_parse_port(rest[0])
        if port is None or port >= 2**16:
            raise ValueError(
                f"The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                "between 0 and 65536."
            )
    else:
        port = default_port

    if not re.match(r"^[\w\.:-]+$", host):
        raise ValueError(
            f"The hostname of the rendezvous endpoint '{endpoint}' must be a dot-separated list of "
            "labels, an IPv4 address, or an IPv6 address."
        )

    return host, portdef _parse_rendezvous_config(config_str: str) -> Dict[str, str]:
    """Extract key-value pairs from a rendezvous configuration string.

    Args:
        config_str:
            A string in format <key1>=<value1>,...,<keyN>=<valueN>.
    """
    config: Dict[str, str] = {}

    config_str = config_str.strip()
    if not config_str:
        return config

    key_values = config_str.split(",")
    for kv in key_values:
        key, *values = kv.split("=", 1)

        key = key.strip()
        if not key:
            raise ValueError(
                "The rendezvous configuration string must be in format "
                "<key1>=<value1>,...,<keyN>=<valueN>."
            )

        value: Optional[str]
        if values:
            value = values[0].strip()
        else:
            value = None
        if not value:
            raise ValueError(
                f"The rendezvous configuration option '{key}' must have a value specified."
            )

        config[key] = value
    return config