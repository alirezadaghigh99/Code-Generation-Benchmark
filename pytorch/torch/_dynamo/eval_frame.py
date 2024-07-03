def optimize(*args, **kwargs):
    def rebuild_ctx():
        return optimize(*args, **kwargs)

    return _optimize(rebuild_ctx, *args, **kwargs)