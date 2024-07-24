class BCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [(a_t - \pi_\theta(s_t))^2]

    Args:
        learning_rate (float): Learing rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        batch_size (int): Mini-batch size.
        policy_type (str): the policy type. Available options are
            ``['deterministic', 'stochastic']``.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
    """

    batch_size: int = 100
    learning_rate: float = 1e-3
    policy_type: str = "deterministic"
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()

    def create(self, device: DeviceArg = False) -> "BC":
        return BC(self, device)

    @staticmethod
    def get_type() -> str:
        return "bc"

class DiscreteBCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm for discrete control.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [-\sum_a p(a|s_t) \log \pi_\theta(a|s_t)]

    where :math:`p(a|s_t)` is implemented as a one-hot vector.

    Args:
        learning_rate (float): Learing rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        batch_size (int): Mini-batch size.
        beta (float): Reguralization factor.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
    """

    batch_size: int = 100
    learning_rate: float = 1e-3
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    beta: float = 0.5

    def create(self, device: DeviceArg = False) -> "DiscreteBC":
        return DiscreteBC(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_bc"

