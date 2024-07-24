class VAEEncoder(nn.Module):  # type: ignore
    _encoder: EncoderWithAction
    _mu: nn.Module
    _logstd: nn.Module
    _min_logstd: float
    _max_logstd: float
    _latent_size: int

    def __init__(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
        latent_size: int,
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
    ):
        super().__init__()
        self._encoder = encoder
        self._mu = nn.Linear(hidden_size, latent_size)
        self._logstd = nn.Linear(hidden_size, latent_size)
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._latent_size = latent_size

    def forward(self, x: TorchObservation, action: torch.Tensor) -> Normal:
        h = self._encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def __call__(self, x: TorchObservation, action: torch.Tensor) -> Normal:
        return super().__call__(x, action)

    @property
    def latent_size(self) -> int:
        return self._latent_size

