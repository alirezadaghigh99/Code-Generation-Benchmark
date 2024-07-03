class _DenseSpatialFilter(nn.Module):
    def __init__(
        self,
        n_chans,
        growth,
        depth,
        in_ch=1,
        bottleneck=4,
        drop_prob=0.0,
        activation=nn.LeakyReLU,
        collapse=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            *[
                _DenseFilter(
                    in_ch + growth * d,
                    growth,
                    bottleneck=bottleneck,
                    drop_prob=drop_prob,
                    activation=activation,
                )
                for d in range(depth)
            ]
        )
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        if collapse:
            self.channel_collapse = _ConvBlock2D(
                n_filters, n_filters, (n_chans, 1), drop_prob=0
            )

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse(x).squeeze(-2)
        return x