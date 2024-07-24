def map_location(device: str) -> Any:
    if "cuda" in device:
        return lambda storage, loc: storage.cuda(device)
    if "cpu" in device:
        return "cpu"
    raise ValueError(f"invalid device={device}")

class View(nn.Module):  # type: ignore
    _shape: Sequence[int]

    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self._shape)

class Swish(nn.Module):  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class GEGLU(nn.Module):  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

