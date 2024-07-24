class DecisionTransformerConfig(TransformerConfig):
    """Config of Decision Transformer.

    Decision Transformer solves decision-making problems as a sequence modeling
    problem.

    References:
        * `Chen at el., Decision Transformer: Reinforcement Learning via
          Sequence Modeling. <https://arxiv.org/abs/2106.01345>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        context_size (int): Prior sequence length.
        max_timestep (int): Maximum environmental timestep.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of attention blocks.
        attn_dropout (float): Dropout probability for attentions.
        resid_dropout (float): Dropout probability for residual connection.
        embed_dropout (float): Dropout probability for embeddings.
        activation_type (str): Type of activation function.
        position_encoding_type (d3rlpy.PositionEncodingType):
            Type of positional encoding (``SIMPLE`` or ``GLOBAL``).
        warmup_steps (int): Warmup steps for learning rate scheduler.
        clip_grad_norm (float): Norm of gradient clipping.
        compile (bool): (experimental) Flag to enable JIT compilation.
    """

    batch_size: int = 64
    learning_rate: float = 1e-4
    encoder_factory: EncoderFactory = make_encoder_field()
    optim_factory: OptimizerFactory = make_optimizer_field()
    num_heads: int = 1
    num_layers: int = 3
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "relu"
    position_encoding_type: PositionEncodingType = PositionEncodingType.SIMPLE
    warmup_steps: int = 10000
    clip_grad_norm: float = 0.25
    compile: bool = False

    def create(self, device: DeviceArg = False) -> "DecisionTransformer":
        return DecisionTransformer(self, device)

    @staticmethod
    def get_type() -> str:
        return "decision_transformer"

