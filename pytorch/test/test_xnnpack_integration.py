        class Linear(torch.nn.Module):
            def __init__(self, weight, bias=None):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return F.linear(x, self.weight, self.bias)        class Linear(torch.nn.Module):
            def __init__(self, weight, bias=None):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return F.linear(x, self.weight, self.bias)        class Linear(torch.nn.Module):
            def __init__(self, weight, bias=None):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return F.linear(x, self.weight, self.bias)