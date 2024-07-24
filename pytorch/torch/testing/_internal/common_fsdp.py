class DummyDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        device: Optional[torch.device] = None,
        *,
        bias: bool = True,
        with_buffer: bool = False,
        dim_multiplier: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device, bias=bias)
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device, bias=bias)
        if with_buffer:
            self.register_buffer("buffer", torch.randn((dim,), device=device))
        else:
            self.buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        if self.buffer is not None:
            z = z + self.buffer
        return z

    def reset_parameters(self):
        if self.buffer is not None:
            torch.nn.init.normal_(self.buffer)

