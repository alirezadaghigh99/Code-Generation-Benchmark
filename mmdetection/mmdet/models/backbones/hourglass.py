class HourglassNet(BaseModule):
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`_ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (Sequence[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (Sequence[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (norm_cfg): Dictionary to construct and config norm layer.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization.

    Example:
        >>> from mmdet.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 downsample_times: int = 5,
                 num_stacks: int = 2,
                 stage_channels: Sequence = (256, 256, 384, 384, 384, 512),
                 stage_blocks: Sequence = (2, 2, 2, 2, 2, 4),
                 feat_channel: int = 256,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 init_cfg: OptMultiConfig = None) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        cur_channel = stage_channels[0]

        self.stem = nn.Sequential(
            ConvModule(
                3, cur_channel // 2, 7, padding=3, stride=2,
                norm_cfg=norm_cfg),
            ResLayer(
                BasicBlock,
                cur_channel // 2,
                cur_channel,
                1,
                stride=2,
                norm_cfg=norm_cfg))

        self.hourglass_modules = nn.ModuleList([
            HourglassModule(downsample_times, stage_channels, stage_blocks)
            for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock,
            cur_channel,
            cur_channel,
            num_stacks - 1,
            norm_cfg=norm_cfg)

        self.conv1x1s = nn.ModuleList([
            ConvModule(
                cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.out_convs = nn.ModuleList([
            ConvModule(
                cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(num_stacks)
        ])

        self.remap_convs = nn.ModuleList([
            ConvModule(
                feat_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self) -> None:
        """Init module weights."""
        # Training Centripetal Model needs to reset parameters for Conv2d
        super().init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward function."""
        inter_feat = self.stem(x)
        out_feats = []

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]

            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)

            if ind < self.num_stacks - 1:
                inter_feat = self.conv1x1s[ind](
                    inter_feat) + self.remap_convs[ind](
                        out_feat)
                inter_feat = self.inters[ind](self.relu(inter_feat))

        return out_feats

