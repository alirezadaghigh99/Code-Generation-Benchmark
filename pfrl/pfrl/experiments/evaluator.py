def run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    with agent.eval_mode():
        return _run_episodes(
            env=env,
            agent=agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )

