def get_current_strategy_limit_usage(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> Optional[int]:
    usage_key = generate_cache_key_for_active_learning_usage(
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )
    value = cache.get(usage_key)
    if value is None:
        return value
    return value[USAGE_KEY]