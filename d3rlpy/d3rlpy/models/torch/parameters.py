class Parameter(nn.Module):  # type: ignore
    _parameter: nn.Parameter

    def __init__(self, data: torch.Tensor):
        super().__init__()
        self._parameter = nn.Parameter(data)

    def forward(self) -> NoReturn:
        raise NotImplementedError(
            "Parameter does not support __call__. Use parameter property instead."
        )

    def __call__(self) -> NoReturn:
        raise NotImplementedError(
            "Parameter does not support __call__. Use parameter property instead."
        )

