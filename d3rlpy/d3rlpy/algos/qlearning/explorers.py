class NormalNoise(Explorer):
    """Normal noise explorer.

    Args:
        mean (float): Mean.
        std (float): Standard deviation.
    """

    _mean: float
    _std: float

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self._mean = mean
        self._std = std

    def sample(
        self, algo: QLearningAlgoProtocol, x: Observation, step: int
    ) -> NDArray:
        """Returns action with noise injection.

        Args:
            algo: Algorithm.
            x: Observation.

        Returns:
            Action with noise injection.
        """
        action = algo.predict(x)
        noise = np.random.normal(self._mean, self._std, size=action.shape)

        minimum: Union[float, NDArray]
        maximum: Union[float, NDArray]
        if isinstance(algo.action_scaler, MinMaxActionScaler):
            # scale noise
            assert algo.action_scaler.minimum is not None
            assert algo.action_scaler.maximum is not None
            minimum = algo.action_scaler.minimum
            maximum = algo.action_scaler.maximum
        else:
            minimum = -1.0
            maximum = 1.0

        return np.clip(action + noise, minimum, maximum)  # type: ignore

class ConstantEpsilonGreedy(Explorer):
    """:math:`\\epsilon`-greedy explorer with constant :math:`\\epsilon`.

    Args:
        epsilon (float): the constant :math:`\\epsilon`.
    """

    _epsilon: float

    def __init__(self, epsilon: float):
        self._epsilon = epsilon

    def sample(
        self, algo: QLearningAlgoProtocol, x: Observation, step: int
    ) -> NDArray:
        action_size = algo.action_size
        assert action_size is not None
        greedy_actions = algo.predict(x)
        batch_size = greedy_actions.shape[0]
        random_actions = np.random.randint(action_size, size=batch_size)
        is_random = np.random.random(batch_size) < self._epsilon
        return np.where(is_random, random_actions, greedy_actions)

