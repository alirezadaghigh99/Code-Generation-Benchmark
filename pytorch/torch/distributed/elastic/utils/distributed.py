def create_c10d_store(
    is_server: bool,
    server_addr: str,
    server_port: int = -1,
    world_size: int = 1,
    timeout: float = (60 * 10),  # 10 min
    wait_for_workers: bool = True,
    retries=3,
    use_libuv: Optional[bool] = None,
):
    if server_port == -1 and world_size > 1:
        raise ValueError(
            f"server_port must be specified when world_size > 1, got server_port={server_port}, world_size={world_size}"
        )

    if server_port != -1:
        logger.info("sever_port: %s, specified, ignoring retries", server_port)

    # only retry when server_port is NOT static
    attempt = retries if server_port == -1 else 1
    while True:
        if server_port != -1:
            port = server_port
        else:
            port = get_free_port()

        logger.info(
            "Creating c10d store on %s:%s\n"
            "  world_size  : %s\n"
            "  is_server   : %s\n"
            "  timeout(sec): %s\n"
            "  use_libuv   : %s\n",
            server_addr,
            port,
            world_size,
            is_server,
            timeout,
            use_libuv,
        )

        try:
            store_builder = functools.partial(
                dist.TCPStore,
                host_name=server_addr,
                port=port,
                world_size=world_size,
                is_master=is_server,
                timeout=datetime.timedelta(seconds=timeout),
                wait_for_workers=wait_for_workers,
            )
            if use_libuv is None:
                # TCPStore default backend may change, don't specify it unless we explicity told to do so.
                store = store_builder()
            else:
                store = store_builder(use_libuv=use_libuv)
            # skips full rank check when we don't have to wait for all workers
            if wait_for_workers:
                _check_full_rank(store, world_size, timeout=timeout)
            logger.info("Successfully created c10d store")
            return store
        except RuntimeError as e:
            # this is brittle, but the underlying exception type is not properly pybinded
            # so we parse the error msg for now, interestingly this is how torch itself
            # detects timeouts and port conflicts in their own unittests
            # see - caffe2/torch/testing/_internal/common_utils.py
            # TODO properly map the exceptions in pybind (c10d/init.cpp)
            if str(e) == _ADDRESS_IN_USE:  # this will only happen on the server
                if attempt < retries:
                    logger.warning(
                        "port: %s already in use, attempt: [%s/%s]",
                        port,
                        attempt,
                        retries,
                    )
                    attempt += 1
                else:
                    raise RuntimeError(
                        f"on {server_addr}, port: {port} already in use"
                    ) from e
            else:
                raise

