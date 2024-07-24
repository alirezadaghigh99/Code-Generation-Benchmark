class AttentionBaseNet(EEGModuleMixin, nn.Module):
    """AttentionBaseNet.

    Neural Network from the paper: EEG motor imagery decoding:
    A framework for comparative analysis with channel attention
    mechanisms

    The paper and original code with more details about the methodological
    choices are available at the [Martin2023]_ and [MartinCode]_.

    The AttentionBaseNet architecture is composed of four modules:
    - Input Block that performs a temporal convolution and a spatial
    convolution.
    - Channel Expansion that modifies the number of channels.
    - An attention block that performs channel attention with several
    options
    - ClassificationHead

    .. versionadded:: 0.9

    Parameters
    ----------

    References
    ----------
    .. [Martin2023] Wimpff, M., Gizzi, L., Zerfowski, J. and Yang, B., 2023.
        EEG motor imagery decoding: A framework for comparative analysis with
        channel attention mechanisms. arXiv preprint arXiv:2310.11198.
    .. [MartinCode] Wimpff, M., Gizzi, L., Zerfowski, J. and Yang, B.
        GitHub https://github.com/martinwimpff/channel-attention (accessed 2024-03-28)
    """

    def __init__(
        self,
        n_times=None,
        n_chans=None,
        n_outputs=None,
        chs_info=None,
        sfreq=None,
        input_window_seconds=None,
        n_temporal_filters: int = 40,
        temp_filter_length_inp: int = 25,
        spatial_expansion: int = 1,
        pool_length_inp: int = 75,
        pool_stride_inp: int = 15,
        dropout_inp: float = 0.5,
        ch_dim: int = 16,
        temp_filter_length: int = 15,
        pool_length: int = 8,
        pool_stride: int = 8,
        dropout: float = 0.5,
        attention_mode: str | None = None,
        reduction_rate: int = 4,
        use_mlp: bool = False,
        freq_idx: int = 0,
        n_codewords: int = 4,
        kernel_size: int = 9,
        extra_params: bool = False,
    ):
        super(AttentionBaseNet, self).__init__()

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds

        self.input_block = _FeatureExtractor(
            n_chans=self.n_chans,
            n_temporal_filters=n_temporal_filters,
            temporal_filter_length=temp_filter_length_inp,
            spatial_expansion=spatial_expansion,
            pool_length=pool_length_inp,
            pool_stride=pool_stride_inp,
            dropout=dropout_inp,
        )

        self.channel_expansion = nn.Sequential(
            nn.Conv2d(
                n_temporal_filters * spatial_expansion, ch_dim, (1, 1), bias=False
            ),
            nn.BatchNorm2d(ch_dim),
            nn.ELU(),
        )

        seq_lengths = self._calculate_sequence_lengths(
            self.n_times,
            [temp_filter_length_inp, temp_filter_length],
            [pool_length_inp, pool_length],
            [pool_stride_inp, pool_stride],
        )

        self.channel_attention_block = _ChannelAttentionBlock(
            attention_mode=attention_mode,
            in_channels=ch_dim,
            temp_filter_length=temp_filter_length,
            pool_length=pool_length,
            pool_stride=pool_stride,
            dropout=dropout,
            reduction_rate=reduction_rate,
            use_mlp=use_mlp,
            seq_len=seq_lengths[0],
            freq_idx=freq_idx,
            n_codewords=n_codewords,
            kernel_size=kernel_size,
            extra_params=extra_params,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_lengths[-1] * ch_dim, self.n_outputs)
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.channel_expansion(x)
        x = self.channel_attention_block(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def _calculate_sequence_lengths(
        input_window_samples: int,
        kernel_lengths: list,
        pool_lengths: list,
        pool_strides: list,
    ):
        seq_lengths = []
        out = input_window_samples
        for k, pl, ps in zip(kernel_lengths, pool_lengths, pool_strides):
            out = np.floor(out + 2 * (k // 2) - k + 1)
            out = np.floor((out - pl) / ps + 1)
            seq_lengths.append(int(out))
        return seq_lengths

