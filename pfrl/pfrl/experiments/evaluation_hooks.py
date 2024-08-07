class OptunaPrunerHook(EvaluationHook):
    """Report evaluation scores to Optuna and prune the trial if necessary.

    Optuna regards trials which raise `optuna.TrialPruned` as unpromissed and
    prune them at the early stages of the training.

    Note that this hook does not support
    pfrl.experiments.train_agent_async.train_agent_async.
    Optuna detects pruning signal by `optuna.TrialPruned` exception, but async training
    mode doesn't re-raise subprocess' exceptions. (See: pfrl.utils.async_.py)

    Args:
        trial (optuna.Trial): Current trial.
    """

    support_train_agent = True
    support_train_agent_batch = True
    support_train_agent_async = False  # unsupported

    def __init__(self, trial):
        if not _optuna_available:
            raise RuntimeError("OptunaPrunerHook requires optuna installed.")
        self.trial = trial

    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats):
        """Call the hook.

        Args:
            env: Environment.
            agent: Agent.
            evaluator: Evaluator.
            step: Current timestep. (Not the number of evaluations so far)
            eval_stats (dict): Last evaluation stats from
                pfrl.experiments.evaluator.eval_performance().
            agent_stats (List of pairs): Last agent stats from
                agent.get_statistics().
            env_stats: Last environment stats from
                env.get_statistics().

        Raises:
            optuna.TrialPruned: Raise when the trial should be pruned immediately.
                Note that you don't need to care about this exception since Optuna will
                catch `optuna.TrialPruned` and stop the trial properly.
        """
        score = eval_stats["mean"]
        self.trial.report(score, step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

