class DiscreteMeanQFunction(DiscreteQFunction):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: TorchObservation) -> QFunctionOutput:
        return QFunctionOutput(
            q_value=self._fc(self._encoder(x)),
            quantiles=None,
            taus=None,
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder

class ContinuousMeanQFunction(ContinuousQFunction):
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, hidden_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> QFunctionOutput:
        return QFunctionOutput(
            q_value=self._fc(self._encoder(x, action)),
            quantiles=None,
            taus=None,
        )

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder

