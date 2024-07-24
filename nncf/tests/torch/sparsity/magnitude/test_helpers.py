class MagnitudeTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 2, 2, 9, -2)
        self.conv2 = create_conv(2, 1, 3, -10, 0)

    def forward(self, x):
        return self.conv2(self.conv1(x))

