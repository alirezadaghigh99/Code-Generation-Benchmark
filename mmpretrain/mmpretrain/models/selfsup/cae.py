class DALLEEncoder(BaseModule):
    """DALL-E Encoder for feature extraction.

    Args:
        group_count (int): Number of groups in DALL-E encoder. Defaults to 4.
        n_hid (int): Dimension of hidden layers. Defaults to 256.
        n_blk_per_group (int): Number of blocks per group. Defaults to 2.
        input_channels: (int): The channels of input images. Defaults to 3.
        vocab_size (int): Vocabulary size, indicating the number of classes.
            Defaults to 8192.
        device (torch.device): Device of parameters. Defaults to
            ``torch.device('cpu')``.
        requires_grad (bool): Require gradient or not. Defaults to False.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 group_count: int = 4,
                 n_hid: int = 256,
                 n_blk_per_group: int = 2,
                 input_channels: int = 3,
                 vocab_size: int = 8192,
                 device: torch.device = torch.device('cpu'),
                 requires_grad: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg=init_cfg)
        self.input_channels = input_channels

        blk_range = range(n_blk_per_group)
        n_layers = group_count * n_blk_per_group
        make_conv = partial(Conv2d, device=device, requires_grad=requires_grad)
        make_blk = partial(
            EncoderBlock,
            n_layers=n_layers,
            device=device,
            requires_grad=requires_grad)

        self.blocks = nn.Sequential(
            OrderedDict([
                ('input', make_conv(input_channels, 1 * n_hid, 7)),
                ('group_1',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i + 1}', make_blk(1 * n_hid, 1 * n_hid))
                           for i in blk_range],
                         ('pool', nn.MaxPool2d(kernel_size=2)),
                     ]))),
                ('group_2',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i + 1}',
                            make_blk(1 * n_hid if i == 0 else 2 * n_hid,
                                     2 * n_hid)) for i in blk_range],
                         ('pool', nn.MaxPool2d(kernel_size=2)),
                     ]))),
                ('group_3',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i + 1}',
                            make_blk(2 * n_hid if i == 0 else 4 * n_hid,
                                     4 * n_hid)) for i in blk_range],
                         ('pool', nn.MaxPool2d(kernel_size=2)),
                     ]))),
                ('group_4',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i + 1}',
                            make_blk(4 * n_hid if i == 0 else 8 * n_hid,
                                     8 * n_hid)) for i in blk_range],
                     ]))),
                ('output',
                 nn.Sequential(
                     OrderedDict([
                         ('relu', nn.ReLU()),
                         ('conv',
                          make_conv(
                              8 * n_hid, vocab_size, 1, use_float16=False)),
                     ]))),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of DALL-E encoder.

        Args:
            x (torch.Tensor): The input images with shape (B, C, H, W).

        Returns:
            torch.Tensor: The output with shape (B, vocab_size, h, w).
        """
        x = x.float()
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model \
                    built for {self.input_channels}')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)

class CAEPretrainViT(BEiTViT):
    """Vision Transformer for CAE pre-training and the implementation is based
    on BEiTViT.

    Args:
        arch (str | dict): Vision Transformer architecture. Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        bias (bool | str): The option to add leanable bias for q, k, v. If bias
            is True, it will add leanable bias. If bias is 'qv_bias', it will
            only add leanable bias for q, v. If bias is False, it will not add
            bias for q, k, v. Default to 'qv_bias'.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            It only works without input mask. Defaults to ``"avg_featmap"``.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        layer_scale_init_value (float, optional): The init value of gamma in
            BEiTTransformerEncoderLayer.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        arch: str = 'b',
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        out_indices: int = -1,
        drop_rate: float = 0,
        drop_path_rate: float = 0,
        bias: bool = 'qv_bias',
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        final_norm: bool = True,
        out_type: str = 'raw',
        frozen_stages: int = -1,
        use_abs_pos_emb: bool = True,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = False,
        layer_scale_init_value: float = None,
        interpolate_mode: str = 'bicubic',
        patch_cfg: dict = dict(),
        layer_cfgs: dict = dict(),
        init_cfg: dict = [
            dict(type='Constant', val=1, layer=['LayerNorm']),
            dict(type='TruncNormal', std=0.02, layer=['Conv2d']),
            dict(type='Xavier', distribution='uniform', layer=['Linear'])
        ]
    ) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            bias=bias,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            out_type=out_type,
            with_cls_token=True,
            frozen_stages=frozen_stages,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            layer_scale_init_value=layer_scale_init_value,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)
        self.pos_embed.requires_grad = False
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # initialize position  embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5),
                self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())

            trunc_normal_(self.cls_token, std=.02)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Generate features for masked images.

        This function generates mask images and get the hidden features for
        visible patches.

        The function supports two kind of forward behaviors. If the ``mask`` is
        not ``None``, the forward function will be executed as masked image
        modeling pre-training; if the ``mask`` is ``None``, the forward
        function will call ``super().forward()``, which extract features from
        images without mask.

        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (torch.Tensor, optional): Mask for input, which is of shape
                B x L.

        Returns:
            torch.Tensor: hidden features.
        """
        if mask is None:
            return super().forward(x)

        else:
            x, _ = self.patch_embed(x)
            batch_size, _, dim = x.size()

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

            # NOTE: unmasked embeddings
            x_unmasked = x[~mask].reshape(batch_size, -1, dim)
            x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=1)

            pos_embed = self.pos_embed.expand(batch_size, self.num_patches + 1,
                                              dim)
            pos_embed_unmasked = pos_embed[:, 1:][~mask].reshape(
                batch_size, -1, dim)
            pos_embed_unmasked = torch.cat(
                (pos_embed[:, :1], pos_embed_unmasked), dim=1)
            x_unmasked = x_unmasked + pos_embed_unmasked

            x_unmasked = self.drop_after_pos(x_unmasked)

            for i, layer in enumerate(self.layers):
                x_unmasked = layer(x=x_unmasked, rel_pos_bias=None)

                if i == len(self.layers) - 1 and self.final_norm:
                    x_unmasked = self.norm1(x_unmasked)

            return x_unmasked

