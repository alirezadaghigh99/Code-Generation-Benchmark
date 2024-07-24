class EEGConformer(EEGModuleMixin, nn.Module):
    """EEG Conformer.

    Convolutional Transformer for EEG decoding.

    The paper and original code with more details about the methodological
    choices are available at the [Song2022]_ and [ConformerCode]_.

    This neural network architecture receives a traditional braindecode input.
    The input shape should be three-dimensional matrix representing the EEG
    signals.

         `(batch_size, n_channels, n_timesteps)`.

    The EEG Conformer architecture is composed of three modules:
        - PatchEmbedding
        - TransformerEncoder
        - ClassificationHead

    Notes
    -----
    The authors recommend using data augmentation before using Conformer,
    e.g. segmentation and recombination,
    Please refer to the original paper and code for more details.

    The model was initially tuned on 4 seconds of 250 Hz data.
    Please adjust the scale of the temporal convolutional layer,
    and the pooling layer for better performance.

    .. versionadded:: 0.8

    We aggregate the parameters based on the parts of the models, or
    when the parameters were used first, e.g. n_filters_time.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.
    att_depth: int
        Number of self-attention layers.
    att_heads: int
        Number of attention heads.
    att_drop_prob: float
        Dropout rate of the self-attention layer.
    final_fc_length: int | str
        The dimension of the fully connected layer.
    return_features: bool
        If True, the forward method returns the features before the
        last classification layer. Defaults to False.
    n_classes :
        Alias for n_outputs.
    n_channels :
        Alias for n_chans.
    input_window_samples :
        Alias for n_times.
    References
    ----------
    .. [Song2022] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering,
       31, pp.710-719. https://ieeexplore.ieee.org/document/9991178
    .. [ConformerCode] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       https://github.com/eeyhsong/EEG-Conformer.
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
        final_fc_length="auto",
        return_features=False,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        n_classes=None,
        n_channels=None,
        input_window_samples=None,
        add_log_softmax=False,
    ):
        n_outputs, n_chans, n_times = deprecated_args(
            self,
            ("n_classes", "n_outputs", n_classes, n_outputs),
            ("n_channels", "n_chans", n_channels, n_chans),
            ("input_window_samples", "n_times", input_window_samples, n_times),
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
        self.mapping = {
            "classification_head.fc.6.weight": "final_layer.final_layer.0.weight",
            "classification_head.fc.6.bias": "final_layer.final_layer.0.bias",
        }

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_classes, n_channels, input_window_samples
        if not (self.n_chans <= 64):
            warnings.warn(
                "This model has only been tested on no more "
                + "than 64 channels. no guarantee to work with "
                + "more channels.",
                UserWarning,
            )

        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=self.n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
        )

        if final_fc_length == "auto":
            assert self.n_times is not None
            final_fc_length = self.get_fc_size()

        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob,
        )

        self.fc = _FullyConnected(final_fc_length=final_fc_length)

        self.final_layer = _FinalLayer(
            n_classes=self.n_outputs,
            return_features=return_features,
            add_log_softmax=self.add_log_softmax,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = self.final_layer(x)
        return x

    def get_fc_size(self):
        out = self.patch_embedding(torch.ones((1, 1, self.n_chans, self.n_times)))
        size_embedding_1 = out.cpu().data.numpy().shape[1]
        size_embedding_2 = out.cpu().data.numpy().shape[2]

        return size_embedding_1 * size_embedding_2

