class ATCNet(EEGModuleMixin, nn.Module):
    """ATCNet model from [1]_

    Pytorch implementation based on official tensorflow code [2]_.

    Parameters
    ----------
    input_window_seconds : float, optional
        Time length of inputs, in seconds. Defaults to 4.5 s, as in BCI-IV 2a
        dataset.
    sfreq : int, optional
        Sampling frequency of the inputs, in Hz. Default to 250 Hz, as in
        BCI-IV 2a dataset.
    conv_block_n_filters : int
        Number temporal filters in the first convolutional layer of the
        convolutional block, denoted F1 in figure 2 of the paper [1]_. Defaults
        to 16 as in [1]_.
    conv_block_kernel_length_1 : int
        Length of temporal filters in the first convolutional layer of the
        convolutional block, denoted Kc in table 1 of the paper [1]_. Defaults
        to 64 as in [1]_.
    conv_block_kernel_length_2 : int
        Length of temporal filters in the last convolutional layer of the
        convolutional block. Defaults to 16 as in [1]_.
    conv_block_pool_size_1 : int
        Length of first average pooling kernel in the convolutional block.
        Defaults to 8 as in [1]_.
    conv_block_pool_size_2 : int
        Length of first average pooling kernel in the convolutional block,
        denoted P2 in table 1 of the paper [1]_. Defaults to 7 as in [1]_.
    conv_block_depth_mult : int
        Depth multiplier of depthwise convolution in the convolutional block,
        denoted D in table 1 of the paper [1]_. Defaults to 2 as in [1]_.
    conv_block_dropout : float
        Dropout probability used in the convolution block, denoted pc in
        table 1 of the paper [1]_. Defaults to 0.3 as in [1]_.
    n_windows : int
        Number of sliding windows, denoted n in [1]_. Defaults to 5 as in [1]_.
    att_head_dim : int
        Embedding dimension used in each self-attention head, denoted dh in
        table 1 of the paper [1]_. Defaults to 8 as in [1]_.
    att_num_heads : int
        Number of attention heads, denoted H in table 1 of the paper [1]_.
        Defaults to 2 as in [1_.
    att_dropout : float
        Dropout probability used in the attention block, denoted pa in table 1
        of the paper [1]_. Defaults to 0.5 as in [1]_.
    tcn_depth : int
        Depth of Temporal Convolutional Network block (i.e. number of TCN
        Residual blocks), denoted L in table 1 of the paper [1]_. Defaults to 2
        as in [1]_.
    tcn_kernel_size : int
        Temporal kernel size used in TCN block, denoted Kt in table 1 of the
        paper [1]_. Defaults to 4 as in [1]_.
    tcn_n_filters : int
        Number of filters used in TCN convolutional layers (Ft). Defaults to
        32 as in [1]_.
    tcn_dropout : float
        Dropout probability used in the TCN block, denoted pt in table 1
        of the paper [1]_. Defaults to 0.3 as in [1]_.
    tcn_activation : torch.nn.Module
        Nonlinear activation to use. Defaults to nn.ELU().
    concat : bool
        When ``True``, concatenates each slidding window embedding before
        feeding it to a fully-connected layer, as done in [1]_. When ``False``,
        maps each slidding window to `n_outputs` logits and average them.
        Defaults to ``False`` contrary to what is reported in [1]_, but
        matching what the official code does [2]_.
    max_norm_const : float
        Maximum L2-norm constraint imposed on weights of the last
        fully-connected layer. Defaults to 0.25.
    n_channels:
        Alias for n_chans.
    n_classes:
        Alias for n_outputs.
    input_size_s:
        Alias for input_window_seconds.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics,
           2022, doi: 10.1109/TII.2022.3197419.
    .. [2] https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        input_window_seconds=None,
        sfreq=250,
        conv_block_n_filters=16,
        conv_block_kernel_length_1=64,
        conv_block_kernel_length_2=16,
        conv_block_pool_size_1=8,
        conv_block_pool_size_2=7,
        conv_block_depth_mult=2,
        conv_block_dropout=0.3,
        n_windows=5,
        att_head_dim=8,
        att_num_heads=2,
        att_dropout=0.5,
        tcn_depth=2,
        tcn_kernel_size=4,
        tcn_n_filters=32,
        tcn_dropout=0.3,
        tcn_activation=nn.ELU(),
        concat=False,
        max_norm_const=0.25,
        chs_info=None,
        n_times=None,
        n_channels=None,
        n_classes=None,
        input_size_s=None,
        add_log_softmax=False,
    ):
        n_chans, n_outputs, input_window_seconds = deprecated_args(
            self,
            ("n_channels", "n_chans", n_channels, n_chans),
            ("n_classes", "n_outputs", n_classes, n_outputs),
            (
                "input_size_s",
                "input_window_seconds",
                input_size_s,
                input_window_seconds,
            ),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_channels, n_classes, input_size_s
        self.conv_block_n_filters = conv_block_n_filters
        self.conv_block_kernel_length_1 = conv_block_kernel_length_1
        self.conv_block_kernel_length_2 = conv_block_kernel_length_2
        self.conv_block_pool_size_1 = conv_block_pool_size_1
        self.conv_block_pool_size_2 = conv_block_pool_size_2
        self.conv_block_depth_mult = conv_block_depth_mult
        self.conv_block_dropout = conv_block_dropout
        self.n_windows = n_windows
        self.att_head_dim = att_head_dim
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.tcn_depth = tcn_depth
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_n_filters = tcn_n_filters
        self.tcn_dropout = tcn_dropout
        self.tcn_activation = tcn_activation
        self.concat = concat
        self.max_norm_const = max_norm_const

        map = dict()
        for w in range(self.n_windows):
            map[f"max_norm_linears.[{w}].weight"] = f"final_layer.[{w}].weight"
            map[f"max_norm_linears.[{w}].bias"] = f"final_layer.[{w}].bias"
        self.mapping = map

        # Check later if we want to keep the Ensure4d. Not sure if we can
        # remove it or replace it with eipsum.
        self.ensuredims = Ensure4d()
        self.dimshuffle = Rearrange("batch C T 1 -> batch 1 T C")

        self.conv_block = _ConvBlock(
            n_channels=self.n_chans,  # input shape: (batch_size, 1, T, C)
            n_filters=conv_block_n_filters,
            kernel_length_1=conv_block_kernel_length_1,
            kernel_length_2=conv_block_kernel_length_2,
            pool_size_1=conv_block_pool_size_1,
            pool_size_2=conv_block_pool_size_2,
            depth_mult=conv_block_depth_mult,
            dropout=conv_block_dropout,
        )

        self.F2 = int(conv_block_depth_mult * conv_block_n_filters)
        self.Tc = int(self.n_times / (conv_block_pool_size_1 * conv_block_pool_size_2))
        self.Tw = self.Tc - self.n_windows + 1

        self.attention_blocks = nn.ModuleList(
            [
                _AttentionBlock(
                    in_shape=self.F2,
                    head_dim=self.att_head_dim,
                    num_heads=att_num_heads,
                    dropout=att_dropout,
                )
                for _ in range(self.n_windows)
            ]
        )

        self.temporal_conv_nets = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        _TCNResidualBlock(
                            in_channels=self.F2,
                            kernel_size=tcn_kernel_size,
                            n_filters=tcn_n_filters,
                            dropout=tcn_dropout,
                            activation=tcn_activation,
                            dilation=2**i,
                        )
                        for i in range(tcn_depth)
                    ]
                )
                for _ in range(self.n_windows)
            ]
        )

        if self.concat:
            self.final_layer = nn.ModuleList(
                [
                    MaxNormLinear(
                        in_features=self.F2 * self.n_windows,
                        out_features=self.n_outputs,
                        max_norm_val=self.max_norm_const,
                    )
                ]
            )
        else:
            self.final_layer = nn.ModuleList(
                [
                    MaxNormLinear(
                        in_features=self.F2,
                        out_features=self.n_outputs,
                        max_norm_val=self.max_norm_const,
                    )
                    for _ in range(self.n_windows)
                ]
            )

        if self.add_log_softmax:
            self.out_fun = nn.LogSoftmax(dim=1)
        else:
            self.out_fun = nn.Identity()

    def forward(self, X):
        # Dimension: (batch_size, C, T)
        X = self.ensuredims(X)
        # Dimension: (batch_size, C, T, 1)
        X = self.dimshuffle(X)
        # Dimension: (batch_size, 1, T, C)

        # ----- Sliding window -----
        conv_feat = self.conv_block(X)
        # Dimension: (batch_size, F2, Tc, 1)
        conv_feat = conv_feat.view(-1, self.F2, self.Tc)
        # Dimension: (batch_size, F2, Tc)

        # ----- Sliding window -----
        sw_concat = []  # to store sliding window outputs
        for w in range(self.n_windows):
            conv_feat_w = conv_feat[..., w : w + self.Tw]
            # Dimension: (batch_size, F2, Tw)

            # ----- Attention block -----
            att_feat = self.attention_blocks[w](conv_feat_w)
            # Dimension: (batch_size, F2, Tw)

            # ----- Temporal convolutional network (TCN) -----
            tcn_feat = self.temporal_conv_nets[w](att_feat)[..., -1]
            # Dimension: (batch_size, F2)

            # Outputs of sliding window can be either averaged after being
            # mapped by dense layer or concatenated then mapped by a dense
            # layer
            if not self.concat:
                tcn_feat = self.final_layer[w](tcn_feat)

            sw_concat.append(tcn_feat)

        # ----- Aggregation and prediction -----
        if self.concat:
            sw_concat = torch.cat(sw_concat, dim=1)
            sw_concat = self.final_layer[0](sw_concat)
        else:
            if len(sw_concat) > 1:  # more than one window
                sw_concat = torch.stack(sw_concat, dim=0)
                sw_concat = torch.mean(sw_concat, dim=0)
            else:  # one window (# windows = 1)
                sw_concat = sw_concat[0]

        return self.out_fun(sw_concat)

