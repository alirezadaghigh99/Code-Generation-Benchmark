def lock_limits(
    cache: BaseCache,
    workspace: str,
    project: str,
) -> Generator[Union[threading.Lock, redis.lock.Lock], None, None]:
    limits_lock_key = generate_cache_key_for_active_learning_usage_lock(
        workspace=workspace,
        project=project,
    )
    with cache.lock(key=limits_lock_key, expire=MAX_LOCK_TIME) as lock:
        yield lock