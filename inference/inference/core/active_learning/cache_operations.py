def return_strategy_limit_usage_credit(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> None:
    current_value = get_current_strategy_limit_usage(
        cache=cache,
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )
    if current_value is None:
        return None
    current_value = max(current_value - 1, 0)
    set_current_strategy_limit_usage(
        current_value=current_value,
        cache=cache,
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )

def consume_strategy_limit_usage_credit(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> None:
    current_value = get_current_strategy_limit_usage(
        cache=cache,
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )
    if current_value is None:
        current_value = 0
    current_value += 1
    set_current_strategy_limit_usage(
        current_value=current_value,
        cache=cache,
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )

def datapoint_should_be_rejected_based_on_strategy_usage_limits(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    strategy_limits: List[StrategyLimit],
) -> bool:
    for strategy_limit in strategy_limits:
        limit_reached = datapoint_should_be_rejected_based_on_limit_usage(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
            strategy_limit=strategy_limit,
        )
        if limit_reached:
            logger.debug(
                f"Violated Active Learning strategy limit: {strategy_limit.limit_type.name} "
                f"with value {strategy_limit.value} for sampling strategy: {strategy_name}."
            )
            return True
    return False

