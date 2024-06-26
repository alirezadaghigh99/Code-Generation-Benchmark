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