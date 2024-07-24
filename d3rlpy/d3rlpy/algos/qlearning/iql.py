class IQLConfig(LearnableConfig):
    r"""Implicit Q-Learning algorithm.

    IQL is the offline RL algorithm that avoids ever querying values of unseen
    actions while still being able to perform multi-step dynamic programming
    updates.

    There are three functions to train in IQL. First the state-value function
    is trained via expectile regression.

    .. math::

        L_V(\psi) = \mathbb{E}_{(s, a) \sim D}
            [L_2^\tau (Q_\theta (s, a) - V_\psi (s))]

    where :math:`L_2^\tau (u) = |\tau - \mathbb{1}(u < 0)|u^2`.

    The Q-function is trained with the state-value function to avoid query the
    actions.

    .. math::

        L_Q(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}
            [(r + \gamma V_\psi(s') - Q_\theta(s, a))^2]

    Finally, the policy function is trained by using advantage weighted
    regression.

    .. math::

        L_\pi (\phi) = \mathbb{E}_{(s, a) \sim D}
            [\exp(\beta (Q_\theta - V_\psi(s))) \log \pi_\phi(a|s)]

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        value_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the value function.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        expectile (float): Expectile value for value function training.
        weight_temp (float): Inverse temperature value represented as
            :math:`\beta`.
        max_weight (float): Maximum advantage weight value to clip.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    value_encoder_factory: EncoderFactory = make_encoder_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    expectile: float = 0.7
    weight_temp: float = 3.0
    max_weight: float = 100.0

    def create(self, device: DeviceArg = False) -> "IQL":
        return IQL(self, device)

    @staticmethod
    def get_type() -> str:
        return "iql"

