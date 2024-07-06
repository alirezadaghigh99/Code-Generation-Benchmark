def PRNGKey(
    seed: Optional[SeedT] = None, *, root: int = 0, comm=MPI_jax_comm
) -> PRNGKeyT:
    """
    Initialises a PRNGKey using an optional starting seed.
    The same seed will be distributed to all processes.
    """
    if seed is None:
        seed = random_seed()

    if isinstance(seed, int):
        # We can't sync the PRNGKey, so we can only sinc integer seeds
        # see https://github.com/google/jax/pull/16511
        if config.netket_experimental_sharding and jax.process_count() > 1:
            # TODO: use stable jax function
            from jax.experimental import multihost_utils

            seed = int(
                multihost_utils.broadcast_one_to_all(
                    seed, is_source=jax.process_index() == root
                ).item()
            )

        key = jax.random.PRNGKey(seed)
    else:
        key = seed

    if not config.netket_experimental_sharding:
        key = jax.tree_util.tree_map(
            lambda k: mpi.mpi_bcast_jax(k, root=root, comm=comm)[0], key
        )
    return key

