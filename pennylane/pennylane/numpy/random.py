    def default_rng(seed=None):
        # Mostly copied from NumPy, but uses our Generator instead

        if hasattr(seed, "capsule"):  # I changed this line
            # We were passed a BitGenerator, so just wrap it up.
            return Generator(seed)
        if isinstance(seed, Generator):
            # Pass through a Generator.
            return seed
        # Otherwise we need to instantiate a new BitGenerator and Generator as
        # normal.
        return Generator(PCG64(seed))