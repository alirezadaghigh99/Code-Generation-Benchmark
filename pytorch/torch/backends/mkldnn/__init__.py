def flags(enabled=False, deterministic=False):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, deterministic)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)

