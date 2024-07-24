class UNet2(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=False,
        residual=False, activation_type="relu", use_bn=True
    ):
        super(UNet2, self).__init__()

        if activation_type == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_type == "relu":
            activation = nn.ReLU(inplace=True)
        else:
            raise TypeError

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96*2, activation, use_bn=use_bn)
        self.down2 = Down(96*2, 96*4, activation, use_bn=use_bn)

        self.up1 = Up(96*4, 96*2, activation, use_bn=use_bn)
        self.up2 = Up(96*2, 96*1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        if self.residual:
            x += input
        return x

