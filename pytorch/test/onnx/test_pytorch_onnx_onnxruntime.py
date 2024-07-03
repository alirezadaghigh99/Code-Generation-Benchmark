        class Size(torch.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape + y.shape)        class Pad(torch.nn.Module):
            def forward(self, x, pad: List[int]):
                return torch.nn.functional.pad(x, pad)        class Data(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.new_zeros(x.data.size())        class Size(torch.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape + y.shape)        class Pad(torch.nn.Module):
            def forward(self, x, pad: List[int]):
                return torch.nn.functional.pad(x, pad)        class Data(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.new_zeros(x.data.size())