class LazyLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x) -> torch.Tensor:
        if getattr(self, "weight", None) is None:
            self.weight = torch.nn.Parameter(
                torch.empty((self.out_features, self.in_features))
            )
            self.bias = torch.nn.Parameter(torch.empty(self.out_features))

        return torch.nn.functional.linear(x, self.weight, self.bias)

