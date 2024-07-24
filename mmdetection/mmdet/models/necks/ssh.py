class SSH(BaseModule):
    """`SSH Neck` used in `SSH: Single Stage Headless Face Detector.

    <https://arxiv.org/pdf/1708.03979.pdf>`_.

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (list[int]): The number of input channels per scale.
        out_channels (list[int]): The number of output channels  per scale.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN').
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [8, 16, 32, 64]
        >>> out_channels = [16, 32, 64, 128]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = SSH(num_scales=4, in_channels=in_channels,
        ...           out_channels=out_channels)
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 16, 340, 340])
        outputs[1].shape = torch.Size([1, 32, 170, 170])
        outputs[2].shape = torch.Size([1, 64, 84, 84])
        outputs[3].shape = torch.Size([1, 128, 43, 43])
    """

    def __init__(self,
                 num_scales: int,
                 in_channels: List[int],
                 out_channels: List[int],
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 init_cfg: OptMultiConfig = dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        for idx in range(self.num_scales):
            in_c, out_c = self.in_channels[idx], self.out_channels[idx]
            self.add_module(
                f'ssh_module{idx}',
                SSHDetModule(
                    in_channels=in_c,
                    out_channels=out_c,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

    def forward(self, inputs: Tuple[torch.Tensor]) -> tuple:
        assert len(inputs) == self.num_scales

        outs = []
        for idx, x in enumerate(inputs):
            ssh_module = getattr(self, f'ssh_module{idx}')
            out = ssh_module(x)
            outs.append(out)

        return tuple(outs)

