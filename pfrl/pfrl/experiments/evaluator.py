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

class Evaluator(object):
    """Object that is responsible for evaluating a given agent.

    Args:
        agent (Agent): Agent to evaluate.
        env (Env): Env to evaluate the agent on.
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean of returns in evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
    """

    def __init__(
        self,
        agent,
        env,
        n_steps,
        n_episodes,
        eval_interval,
        outdir,
        max_episode_len=None,
        step_offset=0,
        evaluation_hooks=(),
        save_best_so_far_agent=True,
        logger=None,
        use_tensorboard=False,
    ):
        assert (n_steps is None) != (n_episodes is None), (
            "One of n_steps or n_episodes must be None. "
            + "Either we evaluate for a specified number "
            + "of episodes or for a specified number of timesteps."
        )
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.use_tensorboard = use_tensorboard
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.prev_eval_t = self.step_offset - self.step_offset % self.eval_interval
        self.evaluation_hooks = evaluation_hooks
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)
        self.env_get_stats = getattr(self.env, "get_statistics", lambda: [])
        self.env_clear_stats = getattr(self.env, "clear_statistics", lambda: None)
        assert callable(self.env_get_stats)
        assert callable(self.env_clear_stats)

        # Write a header line first
        write_header(self.outdir, self.agent, self.env)

        if use_tensorboard:
            self.tb_writer = create_tb_writer(outdir)

    def evaluate_and_update_max_score(self, t, episodes):
        self.env_clear_stats()
        eval_stats = eval_performance(
            self.env,
            self.agent,
            self.n_steps,
            self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger,
        )
        elapsed = time.time() - self.start_time
        agent_stats = self.agent.get_statistics()
        custom_values = tuple(tup[1] for tup in agent_stats)
        env_stats = self.env_get_stats()
        custom_env_values = tuple(tup[1] for tup in env_stats)
        mean = eval_stats["mean"]
        values = (
            (
                t,
                episodes,
                elapsed,
                mean,
                eval_stats["median"],
                eval_stats["stdev"],
                eval_stats["max"],
                eval_stats["min"],
            )
            + custom_values
            + custom_env_values
        )
        record_stats(self.outdir, values)

        if self.use_tensorboard:
            record_tb_stats(self.tb_writer, agent_stats, eval_stats, env_stats, t)

        for hook in self.evaluation_hooks:
            hook(
                env=self.env,
                agent=self.agent,
                evaluator=self,
                step=t,
                eval_stats=eval_stats,
                agent_stats=agent_stats,
                env_stats=env_stats,
            )

        if mean > self.max_score:
            self.logger.info("The best score is updated %s -> %s", self.max_score, mean)
            self.max_score = mean
            if self.save_best_so_far_agent:
                save_agent(self.agent, "best", self.outdir, self.logger)
        return mean

    def evaluate_if_necessary(self, t, episodes):
        if t >= self.prev_eval_t + self.eval_interval:
            score = self.evaluate_and_update_max_score(t, episodes)
            self.prev_eval_t = t - t % self.eval_interval
            return score
        return None

class AsyncEvaluator(object):
    """Object that is responsible for evaluating asynchronous multiple agents.

    Args:
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
    """

    def __init__(
        self,
        n_steps,
        n_episodes,
        eval_interval,
        outdir,
        max_episode_len=None,
        step_offset=0,
        evaluation_hooks=(),
        save_best_so_far_agent=True,
        logger=None,
    ):
        assert (n_steps is None) != (n_episodes is None), (
            "One of n_steps or n_episodes must be None. "
            + "Either we evaluate for a specified number "
            + "of episodes or for a specified number of timesteps."
        )
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.evaluation_hooks = evaluation_hooks
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)

        # Values below are shared among processes
        self.prev_eval_t = mp.Value(
            "l", self.step_offset - self.step_offset % self.eval_interval
        )
        self._max_score = mp.Value("f", np.finfo(np.float32).min)
        self.wrote_header = mp.Value("b", False)

        # Create scores.txt
        with open(os.path.join(self.outdir, "scores.txt"), "a"):
            pass

        self.record_tb_stats_queue = None
        self.record_tb_stats_thread = None

    @property
    def max_score(self):
        with self._max_score.get_lock():
            v = self._max_score.value
        return v

    def evaluate_and_update_max_score(self, t, episodes, env, agent):
        env_get_stats = getattr(env, "get_statistics", lambda: [])
        env_clear_stats = getattr(env, "clear_statistics", lambda: None)
        assert callable(env_get_stats)
        assert callable(env_clear_stats)
        env_clear_stats()
        eval_stats = eval_performance(
            env,
            agent,
            self.n_steps,
            self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger,
        )
        elapsed = time.time() - self.start_time
        agent_stats = agent.get_statistics()
        custom_values = tuple(tup[1] for tup in agent_stats)
        env_stats = env_get_stats()
        custom_env_values = tuple(tup[1] for tup in env_stats)
        mean = eval_stats["mean"]
        values = (
            (
                t,
                episodes,
                elapsed,
                mean,
                eval_stats["median"],
                eval_stats["stdev"],
                eval_stats["max"],
                eval_stats["min"],
            )
            + custom_values
            + custom_env_values
        )
        record_stats(self.outdir, values)

        if self.record_tb_stats_queue is not None:
            self.record_tb_stats_queue.put([agent_stats, eval_stats, env_stats, t])

        for hook in self.evaluation_hooks:
            hook(
                env=env,
                agent=agent,
                evaluator=self,
                step=t,
                eval_stats=eval_stats,
                agent_stats=agent_stats,
                env_stats=env_stats,
            )

        with self._max_score.get_lock():
            if mean > self._max_score.value:
                self.logger.info(
                    "The best score is updated %s -> %s", self._max_score.value, mean
                )
                self._max_score.value = mean
                if self.save_best_so_far_agent:
                    save_agent(agent, "best", self.outdir, self.logger)
        return mean

    def evaluate_if_necessary(self, t, episodes, env, agent):
        necessary = False
        with self.prev_eval_t.get_lock():
            if t >= self.prev_eval_t.value + self.eval_interval:
                necessary = True
                self.prev_eval_t.value += self.eval_interval
        if necessary:
            with self.wrote_header.get_lock():
                if not self.wrote_header.value:
                    write_header(self.outdir, agent, env)
                    self.wrote_header.value = True
            return self.evaluate_and_update_max_score(t, episodes, env, agent)
        return None

    def start_tensorboard_writer(self, outdir, stop_event):
        self.record_tb_stats_queue = mp.Queue()
        self.record_tb_stats_thread = pfrl.utils.StoppableThread(
            target=record_tb_stats_loop,
            args=[outdir, self.record_tb_stats_queue, stop_event],
            stop_event=stop_event,
        )
        self.record_tb_stats_thread.start()

    def join_tensorboard_writer(self):
        self.record_tb_stats_thread.join()

