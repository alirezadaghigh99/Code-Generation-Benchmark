class TridentBottleneck(Bottleneck):
    """BottleBlock for TridentResNet.

    Args:
        trident_dilations (tuple[int, int, int]): Dilations of different
            trident branch.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
        concat_output (bool): Whether to concat the output list to a Tensor.
            `True` only in the last Block.
    """

    def __init__(self, trident_dilations, test_branch_idx, concat_output,
                 **kwargs):

        super(TridentBottleneck, self).__init__(**kwargs)
        self.trident_dilations = trident_dilations
        self.num_branch = len(trident_dilations)
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx
        self.conv2 = TridentConv(
            self.planes,
            self.planes,
            kernel_size=3,
            stride=self.conv2_stride,
            bias=False,
            trident_dilations=self.trident_dilations,
            test_branch_idx=test_branch_idx,
            init_cfg=dict(
                type='Kaiming',
                distribution='uniform',
                mode='fan_in',
                override=dict(name='conv2')))

    def forward(self, x):

        def _inner_forward(x):
            num_branch = (
                self.num_branch
                if self.training or self.test_branch_idx == -1 else 1)
            identity = x
            if not isinstance(x, list):
                x = (x, ) * num_branch
                identity = x
                if self.downsample is not None:
                    identity = [self.downsample(b) for b in x]

            out = [self.conv1(b) for b in x]
            out = [self.norm1(b) for b in out]
            out = [self.relu(b) for b in out]

            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k],
                                                 self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = [self.norm2(b) for b in out]
            out = [self.relu(b) for b in out]
            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k],
                                                 self.after_conv2_plugin_names)

            out = [self.conv3(b) for b in out]
            out = [self.norm3(b) for b in out]

            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k],
                                                 self.after_conv3_plugin_names)

            out = [
                out_b + identity_b for out_b, identity_b in zip(out, identity)
            ]
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = [self.relu(b) for b in out]
        if self.concat_output:
            out = torch.cat(out, dim=0)
        return out

class TridentResNet(ResNet):
    """The stem layer, stage 1 and stage 2 in Trident ResNet are identical to
    ResNet, while in stage 3, Trident BottleBlock is utilized to replace the
    normal BottleBlock to yield trident output. Different branch shares the
    convolution weight but uses different dilations to achieve multi-scale
    output.

                               / stage3(b0) \
    x - stem - stage1 - stage2 - stage3(b1) - output
                               \ stage3(b2) /

    Args:
        depth (int): Depth of resnet, from {50, 101, 152}.
        num_branch (int): Number of branches in TridentNet.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
        trident_dilations (tuple[int]): Dilations of different trident branch.
            len(trident_dilations) should be equal to num_branch.
    """  # noqa

    def __init__(self, depth, num_branch, test_branch_idx, trident_dilations,
                 **kwargs):

        assert num_branch == len(trident_dilations)
        assert depth in (50, 101, 152)
        super(TridentResNet, self).__init__(depth, **kwargs)
        assert self.num_stages == 3
        self.test_branch_idx = test_branch_idx
        self.num_branch = num_branch

        last_stage_idx = self.num_stages - 1
        stride = self.strides[last_stage_idx]
        dilation = trident_dilations
        dcn = self.dcn if self.stage_with_dcn[last_stage_idx] else None
        if self.plugins is not None:
            stage_plugins = self.make_stage_plugins(self.plugins,
                                                    last_stage_idx)
        else:
            stage_plugins = None
        planes = self.base_channels * 2**last_stage_idx
        res_layer = make_trident_res_layer(
            TridentBottleneck,
            inplanes=(self.block.expansion * self.base_channels *
                      2**(last_stage_idx - 1)),
            planes=planes,
            num_blocks=self.stage_blocks[last_stage_idx],
            stride=stride,
            trident_dilations=dilation,
            style=self.style,
            with_cp=self.with_cp,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            dcn=dcn,
            plugins=stage_plugins,
            test_branch_idx=self.test_branch_idx)

        layer_name = f'layer{last_stage_idx + 1}'

        self.__setattr__(layer_name, res_layer)
        self.res_layers.pop(last_stage_idx)
        self.res_layers.insert(last_stage_idx, layer_name)

        self._freeze_stages()

