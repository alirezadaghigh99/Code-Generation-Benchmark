        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(1, 1)
                self.register_buffer('buffer', torch.ones(1))

            def forward(self, x):
                parameters = tuple(self.parameters())
                buffers = tuple(self.buffers())
                return self.l1(x) + self.buffer, parameters, buffers