class DiscreteRandomPolicyConfig(LearnableConfig):
    r"""Random Policy for discrete control algorithm.

    This is designed for data collection and lightweight interaction tests.
    ``fit`` and ``fit_online`` methods will raise exceptions.
    """

    def create(self, device: DeviceArg = False) -> "DiscreteRandomPolicy":  # type: ignore
        return DiscreteRandomPolicy(self)

    @staticmethod
    def get_type() -> str:
        return "discrete_random_policy"

class RandomPolicyConfig(LearnableConfig):
    r"""Random Policy for continuous control algorithm.

    This is designed for data collection and lightweight interaction tests.
    ``fit`` and ``fit_online`` methods will raise exceptions.

    Args:
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        distribution (str): Random distribution. Available options are
            ``['uniform', 'normal']``.
        normal_std (float): Standard deviation of the normal distribution. This
            is only used when ``distribution='normal'``.
    """

    distribution: str = "uniform"
    normal_std: float = 1.0

    def create(self, device: DeviceArg = False) -> "RandomPolicy":  # type: ignore
        return RandomPolicy(self)

    @staticmethod
    def get_type() -> str:
        return "random_policy"

