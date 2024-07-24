class DiscreteQRQFunction(DiscreteQFunction):
    _action_size: int
    _encoder: Encoder
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
        n_quantiles: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(hidden_size, action_size * n_quantiles)

    def forward(self, x: TorchObservation) -> QFunctionOutput:
        quantiles = self._fc(self._encoder(x))
        quantiles = quantiles.view(-1, self._action_size, self._n_quantiles)
        return QFunctionOutput(
            q_value=quantiles.mean(dim=2),
            quantiles=quantiles,
            taus=_make_taus(self._n_quantiles, device=get_device(x)),
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder

class ContinuousQRQFunction(ContinuousQFunction):
    _encoder: EncoderWithAction
    _fc: nn.Linear
    _n_quantiles: int

    def __init__(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
        n_quantiles: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, n_quantiles)
        self._n_quantiles = n_quantiles

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> QFunctionOutput:
        quantiles = self._fc(self._encoder(x, action))
        return QFunctionOutput(
            q_value=quantiles.mean(dim=1, keepdim=True),
            quantiles=quantiles,
            taus=_make_taus(self._n_quantiles, device=get_device(x)),
        )

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder

