def run_on_leader(pg: dist.ProcessGroup, rank: int):
    def callable(func: Callable[..., T]) -> T:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            return invoke_on_rank_and_broadcast_result(pg, rank, func, *args, **kwargs)

        return wrapped

    return callable