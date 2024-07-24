class ContinuousActionDiffEvaluator(EvaluatorProtocol):
    r"""Returns squared difference of actions between algorithm and dataset.

    This metric suggests how different the greedy-policy is from the given
    episodes in continuous action-space.
    If the given episodes are near-optimal, the small action difference would
    be better.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D} [(a_t - \pi_\phi (s_t))^2]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        total_diffs = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                diff = ((batch.actions - actions) ** 2).sum(axis=1).tolist()
                total_diffs += diff
        return float(np.mean(total_diffs))

class DiscreteActionMatchEvaluator(EvaluatorProtocol):
    r"""Returns percentage of identical actions between algorithm and dataset.

    This metric suggests how different the greedy-policy is from the given
    episodes in discrete action-space.
    If the given episdoes are near-optimal, the large percentage would be
    better.

    .. math::

        \frac{1}{N} \sum^N \parallel
            \{a_t = \text{argmax}_a Q_\theta (s_t, a)\}

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        total_matches = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                match = (batch.actions.reshape(-1) == actions).tolist()
                total_matches += match
        return float(np.mean(total_matches))

class CompareContinuousActionDiffEvaluator(EvaluatorProtocol):
    r"""Action difference between algorithms.

    This metric suggests how different the two algorithms are in continuous
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D}
            [(\pi_{\phi_1}(s_t) - \pi_{\phi_2}(s_t))^2]

    Args:
        base_algo: Target algorithm to comapre with.
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _base_algo: QLearningAlgoProtocol
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self,
        base_algo: QLearningAlgoProtocol,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._base_algo = base_algo
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        total_diffs = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                base_actions = self._base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                diff = ((actions - base_actions) ** 2).sum(axis=1).tolist()
                total_diffs += diff
        return float(np.mean(total_diffs))

class CompareDiscreteActionMatchEvaluator(EvaluatorProtocol):
    r"""Action matches between algorithms.

    This metric suggests how different the two algorithms are in discrete
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D} [\parallel
            \{\text{argmax}_a Q_{\theta_1}(s_t, a)
            = \text{argmax}_a Q_{\theta_2}(s_t, a)\}]

    Args:
        base_algo: Target algorithm to comapre with.
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _base_algo: QLearningAlgoProtocol
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self,
        base_algo: QLearningAlgoProtocol,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._base_algo = base_algo
        self._episodes = episodes

    def __call__(
        self, algo: QLearningAlgoProtocol, dataset: ReplayBufferBase
    ) -> float:
        total_matches = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                base_actions = self._base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                match = (base_actions == actions).tolist()
                total_matches += match
        return float(np.mean(total_matches))

class DiscountedSumOfAdvantageEvaluator(EvaluatorProtocol):
    r"""Returns average of discounted sum of advantage.

    This metric suggests how the greedy-policy selects different actions in
    action-value space.
    If the sum of advantage is small, the policy selects actions with larger
    estimated action-values.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D}
            [\sum_{t' = t} \gamma^{t' - t} A(s_{t'}, a_{t'})]

    where :math:`A(s_t, a_t) = Q_\theta (s_t, a_t)
    - \mathbb{E}_{a \sim \pi} [Q_\theta (s_t, a)]`.

    References:
        * `Murphy., A generalization error for Q-Learning.
          <http://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf>`_

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        total_sums = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                # estimate values for dataset actions
                dataset_values = algo.predict_value(
                    batch.observations, batch.actions
                )

                # estimate values for the current policy
                actions = algo.predict(batch.observations)
                on_policy_values = algo.predict_value(
                    batch.observations, actions
                )

                # calculate advantages
                advantages = (dataset_values - on_policy_values).tolist()

                # calculate discounted sum of advantages
                A = advantages[-1]
                sum_advantages = [A]
                for advantage in reversed(advantages[:-1]):
                    A = advantage + algo.gamma * A
                    sum_advantages.append(A)

                total_sums += sum_advantages
        # smaller is better
        return float(np.mean(total_sums))

class AverageValueEstimationEvaluator(EvaluatorProtocol):
    r"""Returns average value estimation.

    This metric suggests the scale for estimation of Q functions.
    If average value estimation is too large, the Q functions overestimate
    action-values, which possibly makes training failed.

    .. math::

        \mathbb{E}_{s_t \sim D} [ \max_a Q_\theta (s_t, a)]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        total_values = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                values = algo.predict_value(batch.observations, actions)
                total_values += values.tolist()
        return float(np.mean(total_values))

class InitialStateValueEstimationEvaluator(EvaluatorProtocol):
    r"""Returns mean estimated action-values at the initial states.

    This metric suggests how much return the trained policy would get from
    the initial states by deploying the policy to the states.
    If the estimated value is large, the trained policy is expected to get
    higher returns.

    .. math::

        \mathbb{E}_{s_0 \sim D} [Q(s_0, \pi(s_0))]

    References:
        * `Paine et al., Hyperparameter Selection for Offline Reinforcement
          Learning <https://arxiv.org/abs/2007.09055>`_

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBufferBase,
    ) -> float:
        total_values = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                # estimate action-value in initial states
                first_obs = np.expand_dims(batch.observations[0], axis=0)
                actions = algo.predict(first_obs)
                values = algo.predict_value(first_obs, actions)
                total_values.append(values[0])
        return float(np.mean(total_values))

