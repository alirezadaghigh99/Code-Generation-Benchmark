def spawn(func, nprocs=1, **kwargs):
    """
    This function is used to spawn processes for testing.

    Usage:
        # must contains arguments rank, world_size, port
        def do_something(rank, world_size, port):
            ...

        spawn(do_something, nprocs=8)

        # can also pass other arguments
        def do_something(rank, world_size, port, arg1, arg2):
            ...

        spawn(do_something, nprocs=8, arg1=1, arg2=2)

    Args:
        func (Callable): The function to be spawned.
        nprocs (int, optional): The number of processes to spawn. Defaults to 1.
    """
    port = free_port()
    wrapped_func = partial(func, world_size=nprocs, port=port, **kwargs)
    mp.spawn(wrapped_func, nprocs=nprocs)