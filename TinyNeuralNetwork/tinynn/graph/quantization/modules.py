class QGLU(nn.Module):
    def __init__(self, glu: nn.GLU) -> None:
        super().__init__()

        self.dim = glu.dim
        self.f_mul = nnq.FloatFunctional()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        slices = torch.chunk(input, 2, self.dim)
        return self.f_mul.mul(slices[0], self.sigmoid(slices[1]))