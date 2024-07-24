class Size(torch.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape + y.shape)

class Pad(torch.nn.Module):
            def forward(self, x, pad: List[int]):
                return torch.nn.functional.pad(x, pad)

class Data(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.new_zeros(x.data.size())

class Unfold(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.nn.functional.unfold(
                        input, kernel_size=(10, 15), dilation=2, padding=5, stride=3
                    ),
                    torch.nn.functional.unfold(
                        input, kernel_size=(2, 2), dilation=1, padding=0, stride=3
                    ),
                    torch.nn.functional.unfold(
                        input, kernel_size=(1, 1), dilation=5, padding=2, stride=3
                    ),
                )

class Bernoulli(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.bernoulli(x).size(0))

class PixelUnshuffle(torch.nn.Module):
            def forward(self, x):
                return torch.pixel_unshuffle(x, downscale_factor=2)

