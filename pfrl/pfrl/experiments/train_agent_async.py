def train_loop(
    process_idx,
    env,
    agent,
    steps,
    outdir,
    counter,
    episodes_counter,
    stop_event,
    exception_event,
    max_episode_len=None,
    evaluator=None,
    eval_env=None,
    successful_score=None,
    logger=None,
    global_step_hooks=[],
):
    logger = logger or logging.getLogger(__name__)

    if eval_env is None:
        eval_env = env

    def save_model():
        if process_idx == 0:
            # Save the current model before being killed
            dirname = os.path.join(outdir, "{}_except".format(global_t))
            agent.save(dirname)
            logger.info("Saved the current model to %s", dirname)

    try:
        episode_r = 0
        global_t = 0
        local_t = 0
        global_episodes = 0
        obs = env.reset()
        episode_len = 0
        successful = False

        while True:
            # a_t
            a = agent.act(obs)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(a)
            local_t += 1
            episode_r += r
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value

            for hook in global_step_hooks:
                hook(env, agent, global_t)

            if done or reset or global_t >= steps or stop_event.is_set():
                if process_idx == 0:
                    logger.info(
                        "outdir:%s global_step:%s local_step:%s R:%s",
                        outdir,
                        global_t,
                        local_t,
                        episode_r,
                    )
                    logger.info("statistics:%s", agent.get_statistics())

                # Evaluate the current agent
                if evaluator is not None:
                    eval_score = evaluator.evaluate_if_necessary(
                        t=global_t, episodes=global_episodes, env=eval_env, agent=agent
                    )

                    if (
                        eval_score is not None
                        and successful_score is not None
                        and eval_score >= successful_score
                    ):
                        stop_event.set()
                        successful = True
                        # Break immediately in order to avoid an additional
                        # call of agent.act_and_train
                        break

                with episodes_counter.get_lock():
                    episodes_counter.value += 1
                    global_episodes = episodes_counter.value

                if global_t >= steps or stop_event.is_set():
                    break

                # Start a new episode
                episode_r = 0
                episode_len = 0
                obs = env.reset()

            if process_idx == 0 and exception_event.is_set():
                logger.exception("An exception detected, exiting")
                save_model()
                kill_all()

    except (Exception, KeyboardInterrupt):
        save_model()
        raise

    if global_t == steps:
        # Save the final model
        dirname = os.path.join(outdir, "{}_finish".format(steps))
        agent.save(dirname)
        logger.info("Saved the final agent to %s", dirname)

    if successful:
        # Save the successful model
        dirname = os.path.join(outdir, "successful")
        agent.save(dirname)
        logger.info("Saved the successful agent to %s", dirname)