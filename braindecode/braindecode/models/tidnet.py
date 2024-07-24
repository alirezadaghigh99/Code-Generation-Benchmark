class _DenseSpatialFilter(nn.Module):
    def __init__(
        self,
        n_chans,
        growth,
        depth,
        in_ch=1,
        bottleneck=4,
        drop_prob=0.0,
        activation=nn.LeakyReLU,
        collapse=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            *[
                _DenseFilter(
                    in_ch + growth * d,
                    growth,
                    bottleneck=bottleneck,
                    drop_prob=drop_prob,
                    activation=activation,
                )
                for d in range(depth)
            ]
        )
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        if collapse:
            self.channel_collapse = _ConvBlock2D(
                n_filters, n_filters, (n_chans, 1), drop_prob=0
            )

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse(x).squeeze(-2)
        return x

class TIDNet(EEGModuleMixin, nn.Module):
    """Thinker Invariance DenseNet model from Kostas et al 2020.

    See [TIDNet]_ for details.

    Parameters
    ----------
    s_growth : int
        DenseNet-style growth factor (added filters per DenseFilter)
    t_filters : int
        Number of temporal filters.
    drop_prob : float
        Dropout probability
    pooling : int
        Max temporal pooling (width and stride)
    temp_layers : int
        Number of temporal layers
    spat_layers : int
        Number of DenseFilters
    temp_span : float
        Percentage of n_times that defines the temporal filter length:
        temp_len = ceil(temp_span * n_times)
        e.g A value of 0.05 for temp_span with 1500 n_times will yield a temporal
        filter of length 75.
    bottleneck : int
        Bottleneck factor within Densefilter
    summary : int
        Output size of AdaptiveAvgPool1D layer. If set to -1, value will be calculated
        automatically (n_times // pooling).
    in_chans :
        Alias for n_chans.
    n_classes:
        Alias for n_outputs.
    input_window_samples :
        Alias for n_times.

    Notes
    -----
    Code adapted from: https://github.com/SPOClab-ca/ThinkerInvariance/

    References
    ----------
    .. [TIDNet] Kostas, D. & Rudzicz, F.
        Thinker invariance: enabling deep neural networks for BCI across more
        people.
        J. Neural Eng. 17, 056008 (2020).
        doi: 10.1088/1741-2552/abb7a7.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        in_chans=None,
        n_classes=None,
        input_window_samples=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        s_growth=24,
        t_filters=32,
        drop_prob=0.4,
        pooling=15,
        temp_layers=2,
        spat_layers=2,
        temp_span=0.05,
        bottleneck=3,
        summary=-1,
        add_log_softmax=False,
    ):
        n_chans, n_outputs, n_times = deprecated_args(
            self,
            ("in_chans", "n_chans", in_chans, n_chans),
            ("n_classes", "n_outputs", n_classes, n_outputs),
            ("input_window_samples", "n_times", input_window_samples, n_times),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            chs_info=chs_info,
            add_log_softmax=add_log_softmax,
        )
        del n_outputs, n_chans, n_times, input_window_seconds, sfreq, chs_info
        del in_chans, n_classes, input_window_samples

        self.mapping = {
            "classify.1.weight": "final_layer.0.weight",
            "classify.1.bias": "final_layer.0.bias",
        }

        self.temp_len = ceil(temp_span * self.n_times)

        self.dscnn = _TIDNetFeatures(
            s_growth=s_growth,
            t_filters=t_filters,
            n_chans=self.n_chans,
            n_times=self.n_times,
            drop_prob=drop_prob,
            pooling=pooling,
            temp_layers=temp_layers,
            spat_layers=spat_layers,
            temp_span=temp_span,
            bottleneck=bottleneck,
            summary=summary,
        )

        self._num_features = self.dscnn.num_features

        self.flatten = nn.Flatten(start_dim=1)

        self.final_layer = self._create_classifier(self.num_features, self.n_outputs)

    def _create_classifier(self, incoming, n_outputs):
        classifier = nn.Linear(incoming, n_outputs)
        init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        seq_clf = nn.Sequential(
            classifier, nn.LogSoftmax(dim=-1) if self.add_log_softmax else nn.Identity()
        )

        return seq_clf

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        x = self.dscnn(x)
        x = self.flatten(x)
        return self.final_layer(x)

    @property
    def num_features(self):
        return self._num_features

