class DetectoRS_ResNet(ResNet):
    """ResNet backbone for DetectoRS.

    Args:
        sac (dict, optional): Dictionary to construct SAC (Switchable Atrous
            Convolution). Default: None.
        stage_with_sac (list): Which stage to use sac. Default: (False, False,
            False, False).
        rfp_inplanes (int, optional): The number of channels from RFP.
            Default: None. If specified, an additional conv layer will be
            added for ``rfp_feat``. Otherwise, the structure is the same as
            base class.
        output_img (bool): If ``True``, the input image will be inserted into
            the starting position of output. Default: False.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 sac=None,
                 stage_with_sac=(False, False, False, False),
                 rfp_inplanes=None,
                 output_img=False,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        self.pretrained = pretrained
        if init_cfg is not None:
            assert isinstance(init_cfg, dict), \
                f'init_cfg must be a dict, but got {type(init_cfg)}'
            if 'type' in init_cfg:
                assert init_cfg.get('type') == 'Pretrained', \
                    'Only can initialize module by loading a pretrained model'
            else:
                raise KeyError('`init_cfg` must contain the key "type"')
            self.pretrained = init_cfg.get('checkpoint')
        self.sac = sac
        self.stage_with_sac = stage_with_sac
        self.rfp_inplanes = rfp_inplanes
        self.output_img = output_img
        super(DetectoRS_ResNet, self).__init__(**kwargs)

        self.inplanes = self.stem_channels
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            sac = self.sac if self.stage_with_sac[i] else None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                sac=sac,
                rfp_inplanes=rfp_inplanes if i > 0 else None,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

    # In order to be properly initialized by RFP
    def init_weights(self):
        # Calling this method will cause parameter initialization exception
        # super(DetectoRS_ResNet, self).init_weights()

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer`` for DetectoRS."""
        return ResLayer(**kwargs)

    def forward(self, x):
        """Forward function."""
        outs = list(super(DetectoRS_ResNet, self).forward(x))
        if self.output_img:
            outs.insert(0, x)
        return tuple(outs)

    def rfp_forward(self, x, rfp_feats):
        """Forward function for RFP."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            rfp_feat = rfp_feats[i] if i > 0 else None
            for layer in res_layer:
                x = layer.rfp_forward(x, rfp_feat)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

