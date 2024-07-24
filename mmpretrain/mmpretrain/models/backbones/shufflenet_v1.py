class ShuffleUnit(BaseModule):
    """ShuffleUnit block.

    ShuffleNet unit with pointwise group convolution (GConv) and channel
    shuffle.

    Args:
        in_channels (int): The input channels of the ShuffleUnit.
        out_channels (int): The output channels of the ShuffleUnit.
        groups (int): The number of groups to be used in grouped 1x1
            convolutions in each ShuffleUnit. Default: 3
        first_block (bool): Whether it is the first ShuffleUnit of a
            sequential ShuffleUnits. Default: True, which means not using the
            grouped 1x1 convolution.
        combine (str): The ways to combine the input and output
            branches. Default: 'add'.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=3,
                 first_block=True,
                 combine='add',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_block = first_block
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        self.with_cp = with_cp

        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
            assert in_channels == out_channels, (
                'in_channels must be equal to out_channels when combine '
                'is add')
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f'Cannot combine tensors with {self.combine}. '
                             'Only "add" and "concat" are supported')

        self.first_1x1_groups = 1 if first_block else self.groups
        self.g_conv_1x1_compress = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            groups=self.first_1x1_groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.depthwise_conv3x3_bn = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=self.depthwise_stride,
            padding=1,
            groups=self.bottleneck_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.g_conv_1x1_expand = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            groups=self.groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.act = build_activation_layer(act_cfg)

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.g_conv_1x1_compress(x)
            out = self.depthwise_conv3x3_bn(out)

            if self.groups > 1:
                out = channel_shuffle(out, self.groups)

            out = self.g_conv_1x1_expand(out)

            if self.combine == 'concat':
                residual = self.avgpool(residual)
                out = self.act(out)
                out = self._combine_func(residual, out)
            else:
                out = self._combine_func(residual, out)
                out = self.act(out)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

