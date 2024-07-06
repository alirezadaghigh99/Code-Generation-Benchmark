def device_count_per_rank():
    """
    Helper functions which returns the number of jax devices netket will use for every
    MPI rank.

    Returns:
        jax.device_count() if config.netket_experimental_sharding is True, and 1 otherwise
    """
    if config.netket_experimental_sharding:
        if mpi.n_nodes > 1:
            # this should never be triggered as we disable mpi when sharding
            raise NotImplementedError("hybrid mpi and sharding is not not supported")
        return jax.device_count()
    else:  # mpi or serial
        return 1

