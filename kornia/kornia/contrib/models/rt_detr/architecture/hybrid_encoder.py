class HybridEncoder(Module):
    def __init__(self, in_channels: list[int], hidden_dim: int, dim_feedforward: int, expansion: float = 1.0) -> None:
        super().__init__()
        self.input_proj = nn.ModuleList([ConvNormAct(in_ch, hidden_dim, 1, act="none") for in_ch in in_channels])
        self.aifi = AIFI(hidden_dim, 8, dim_feedforward)
        self.ccfm = CCFM(len(in_channels), hidden_dim, expansion)

    def forward(self, fmaps: list[Tensor]) -> list[Tensor]:
        projected_maps = [proj(fmap) for proj, fmap in zip(self.input_proj, fmaps)]
        projected_maps[-1] = self.aifi(projected_maps[-1])
        new_fmaps = self.ccfm(projected_maps)
        return new_fmaps

class RepVggBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, act="none")
        self.conv2 = ConvNormAct(in_channels, out_channels, 1, act="none")
        self.act = nn.SiLU(inplace=True)
        self.conv: Optional[nn.Conv2d] = None

    def forward(self, x: Tensor) -> Tensor:
        if self.conv is not None:
            out = self.act(self.conv(x))
        else:
            out = self.act(self.conv1(x) + self.conv2(x))
        return out

    @torch.no_grad()
    def optimize_for_deployment(self) -> None:
        def _fuse_conv_bn_weights(m: ConvNormAct) -> tuple[nn.Parameter, nn.Parameter]:
            if m.norm.running_mean is None or m.norm.running_var is None:
                raise ValueError

            return fuse_conv_bn_weights(
                m.conv.weight,
                m.conv.bias,
                m.norm.running_mean,
                m.norm.running_var,
                m.norm.eps,
                m.norm.weight,
                m.norm.bias,
            )

        kernel3x3, bias3x3 = _fuse_conv_bn_weights(self.conv1)
        kernel1x1, bias1x1 = _fuse_conv_bn_weights(self.conv2)
        kernel3x3.add_(pad(kernel1x1, [1, 1, 1, 1]))
        bias3x3.add_(bias1x1)

        self.conv = nn.Conv2d(kernel3x3.shape[1], kernel3x3.shape[0], 3, 1, 1)
        self.conv.weight = kernel3x3
        self.conv.bias = bias3x3

