class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        img_size (int | tuple): The size of input image. Default: 224
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None
        conv_cfg (dict, optional): The config dict for conv layers.
            Default: None
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=768,
                 norm_cfg=None,
                 conv_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg)
        warnings.warn('The `PatchEmbed` in mmpretrain will be deprecated. '
                      'Please use `mmcv.cnn.bricks.transformer.PatchEmbed`. '
                      "It's more general and supports dynamic input shape")

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.embed_dims = embed_dims

        # Use conv layer to embed
        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=16, stride=16, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, in_channels, embed_dims)

        # Calculate how many patches a input image is splited to.
        h_out, w_out = [(self.img_size[i] + 2 * self.projection.padding[i] -
                         self.projection.dilation[i] *
                         (self.projection.kernel_size[i] - 1) - 1) //
                        self.projection.stride[i] + 1 for i in range(2)]

        self.patches_resolution = (h_out, w_out)
        self.num_patches = h_out * w_out

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        # The output size is (B, N, D), where N=H*W/P/P, D is embid_dim
        x = self.projection(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x

class PatchMerging(BaseModule):
    """Merge patch feature map.

    Modified from mmcv, and this module supports specifying whether to use
    post-norm.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map ((used in Swin Transformer)). Our
    implementation uses :class:`torch.nn.Unfold` to merge patches, which is
    about 25% faster than the original implementation. However, we need to
    modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels. To gets fully covered
            by filter and stride you specified.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Defaults to None, which means to be set as
            ``kernel_size``.
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Defaults to "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Defaults to 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults to False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        use_post_norm (bool): Whether to use post normalization here.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 use_post_norm=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_post_norm = use_post_norm

        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adaptive_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

        if norm_cfg is not None:
            # build pre or post norm layer based on different channels
            if self.use_post_norm:
                self.norm = build_norm_layer(norm_cfg, out_channels)[1]
            else:
                self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
            H, W = x.shape[-2:]

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)
        x = self.sampler(x)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        if self.use_post_norm:
            # use post-norm here
            x = self.reduction(x)
            x = self.norm(x) if self.norm else x
        else:
            x = self.norm(x) if self.norm else x
            x = self.reduction(x)

        return x, output_size

