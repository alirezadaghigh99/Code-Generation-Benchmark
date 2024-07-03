def flags(
    enabled=False,
    benchmark=False,
    benchmark_limit=10,
    deterministic=False,
    allow_tf32=True,
):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            enabled, benchmark, benchmark_limit, deterministic, allow_tf32
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)def flags(
    enabled=False,
    benchmark=False,
    benchmark_limit=10,
    deterministic=False,
    allow_tf32=True,
):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            enabled, benchmark, benchmark_limit, deterministic, allow_tf32
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)