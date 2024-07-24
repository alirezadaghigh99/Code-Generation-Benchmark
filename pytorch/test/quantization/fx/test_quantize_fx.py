class Conv2d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv2d(*args)

            def forward(self, x):
                return self.conv(x)

class Conv1d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv1d(*args)

            def forward(self, x):
                return self.conv(x)

class Conv3d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv3d(*args)

            def forward(self, x):
                return self.conv(x)

class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(50, 50, 1)

            def forward(self, inputs: torch.Tensor, state: List[torch.Tensor]):
                h = state[0]
                c = state[1]
                return self.lstm(inputs, (h, c))

class Conv2d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv2d(*args)

            def forward(self, x):
                return self.conv(x)

class ConvTranspose2d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose2d(*args)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.deconv(x)
                if use_relu:
                    y = self.relu(y)
                return y

class ConvTranspose3d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose3d(*args)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.deconv(x)
                if use_relu:
                    y = self.relu(y)
                return y

class ConvTranspose1d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose1d(*args)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.deconv(x)
                if use_relu:
                    y = self.relu(y)
                return y

