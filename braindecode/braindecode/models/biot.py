class BIOT(EEGModuleMixin, nn.Module):
    """BIOT: Cross-data Biosignal Learning in the Wild from [Yang2023]_

    BIOT is a large language model for biosignal classification. It is
    a wrapper around the `BIOTEncoder` and `ClassificationHead` modules.

    It is designed for N-dimensional biosignal data such as EEG, ECG, etc.
    The method was proposed by Yang et al. [Yang2023]_ and the code is
    available at [Code2023]_

    The model is trained with a contrastive loss on large EEG datasets
    TUH Abnormal EEG Corpus with 400K samples and Sleep Heart Health Study
    5M. Here, we only provide the model architecture, not the pre-trained
    weights or contrastive loss training.

    The architecture is based on the `LinearAttentionTransformer` and
    `PatchFrequencyEmbedding` modules.
    The `BIOTEncoder` is a transformer that takes the input data and outputs
    a fixed-size representation of the input data. More details are
    present in the `BIOTEncoder` class.

    The `ClassificationHead` is an ELU activation layer, followed by a simple
    linear layer that takes the output of the `BIOTEncoder` and outputs
    the classification probabilities.

    .. versionadded:: 0.9

    Parameters
    ----------
    emb_size : int, optional
        The size of the embedding layer, by default 256
    att_num_heads : int, optional
        The number of attention heads, by default 8
    n_layers : int, optional
        The number of transformer layers, by default 4

    References
    ----------
    .. [Yang2023] Yang, C., Westover, M.B. and Sun, J., 2023, November. BIOT:
       Biosignal Transformer for Cross-data Learning in the Wild. In Thirty-seventh
       Conference on Neural Information Processing Systems, NeurIPS.
    .. [Code2023] Yang, C., Westover, M.B. and Sun, J., 2023. BIOT
       Biosignal Transformer for Cross-data Learning in the Wild.
       GitHub https://github.com/ycq091044/BIOT (accessed 2024-02-13)
    """

    def __init__(
        self,
        emb_size=256,
        att_num_heads=8,
        n_layers=4,
        sfreq=200,
        hop_length=100,
        return_feature=False,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq
        self.emb_size = emb_size
        self.hop_length = hop_length
        self.att_num_heads = att_num_heads
        self.n_layers = n_layers
        self.return_feature = return_feature
        if (self.sfreq != 200) & (self.sfreq is not None):
            warn(
                "This model has only been trained on a dataset with 200 Hz. "
                + "no guarantee to generalize well with the default parameters",
                UserWarning,
            )
        if self.n_chans > emb_size:
            warn(
                "The number of channels is larger than the embedding size. "
                + "This may cause overfitting. Consider using a larger "
                + "embedding size or a smaller number of channels.",
                UserWarning,
            )
        if self.hop_length > self.sfreq:
            warn(
                "The hop length is larger than the sampling frequency. "
                + "This may cause aliasing. Consider using a smaller "
                "hop length.",
                UserWarning,
            )
            hop_length = self.sfreq // 2
        self.encoder = _BIOTEncoder(
            emb_size=emb_size,
            att_num_heads=att_num_heads,
            n_layers=n_layers,
            n_chans=self.n_chans,
            n_fft=self.sfreq,
            hop_length=hop_length,
        )

        self.classifier = _ClassificationHead(
            emb_size=emb_size, n_outputs=self.n_outputs
        )

    def forward(self, x):
        """
        Pass the input through the BIOT encoder, and then through the
        classification head.

        Parameters
        ----------
        x: Tensor
            (batch_size, n_channels, n_times)

        Returns
        -------
        out: Tensor
            (batch_size, n_outputs)
        (out, emb): tuple Tensor
            (batch_size, n_outputs), (batch_size, emb_size)
        """
        emb = self.encoder(x)
        x = self.classifier(emb)

        if self.return_feature:
            return x, emb
        else:
            return x

