def enable_ddp_env(backend="gloo"):
    def _enable_ddp_env(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def find_free_port() -> str:
                s = socket.socket()
                s.bind(("localhost", 0))  # Bind to a free port provided by the host.
                return str(s.getsockname()[1])

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = find_free_port()

            return distributed_worker(
                main_func=func,
                args=args,
                kwargs=kwargs,
                backend=backend,
                init_method="file:///tmp/detectron2go_test_ddp_init_{}".format(
                    uuid.uuid4().hex
                ),
                dist_params=DistributedParams(
                    local_rank=0,
                    machine_rank=0,
                    global_rank=0,
                    num_processes_per_machine=1,
                    world_size=1,
                ),
                return_save_file=None,  # don't save file
            )

        return wrapper

    return _enable_ddp_env

