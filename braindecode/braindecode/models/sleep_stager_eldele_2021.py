class SleepStagerEldele2021(EEGModuleMixin, nn.Module):
    """Sleep Staging Architecture from Eldele et al 2021.

    Attention based Neural Net for sleep staging as described in [Eldele2021]_.
    The code for the paper and this model is also available at [1]_.
    Takes single channel EEG as input.
    Feature extraction module based on multi-resolution convolutional neural network (MRCNN)
    and adaptive feature recalibration (AFR).
    The second module is the temporal context encoder (TCE) that leverages a multi-head attention
    mechanism to capture the temporal dependencies among the extracted features.

    Warning - This model was designed for signals of 30 seconds at 100Hz or 125Hz (in which case
    the reference architecture from [1]_ which was validated on SHHS dataset [2]_ will be used)
    to use any other input is likely to make the model perform in unintended ways.

    Parameters
    ----------
    n_tce : int
        Number of TCE clones.
    d_model : int
        Input dimension for the TCE.
        Also the input dimension of the first FC layer in the feed forward
        and the output of the second FC layer in the same.
        Increase for higher sampling rate/signal length.
        It should be divisible by n_attn_heads
    d_ff : int
        Output dimension of the first FC layer in the feed forward and the
        input dimension of the second FC layer in the same.
    n_attn_heads : int
        Number of attention heads. It should be a factor of d_model
    dropout : float
        Dropout rate in the PositionWiseFeedforward layer and the TCE layers.
    after_reduced_cnn_size : int
        Number of output channels produced by the convolution in the AFR module.
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    n_classes : int
        Alias for `n_outputs`.
    input_size_s : float
        Alias for `input_window_seconds`.

    References
    ----------
    .. [Eldele2021] E. Eldele et al., "An Attention-Based Deep Learning Approach for Sleep Stage
        Classification With Single-Channel EEG," in IEEE Transactions on Neural Systems and
        Rehabilitation Engineering, vol. 29, pp. 809-818, 2021, doi: 10.1109/TNSRE.2021.3076234.

    .. [1] https://github.com/emadeldeen24/AttnSleep

    .. [2] https://sleepdata.org/datasets/shhs
    """

    def __init__(
        self,
        sfreq=None,
        n_tce=2,
        d_model=80,
        d_ff=120,
        n_attn_heads=5,
        dropout=0.1,
        input_window_seconds=None,
        n_outputs=None,
        after_reduced_cnn_size=30,
        return_feats=False,
        chs_info=None,
        n_chans=None,
        n_times=None,
        n_classes=None,
        input_size_s=None,
    ):
        (
            n_outputs,
            input_window_seconds,
        ) = deprecated_args(
            self,
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
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_classes, input_size_s

        self.mapping = {
            "fc.weight": "final_layer.weight",
            "fc.bias": "final_layer.bias",
        }

        if not (
            (self.input_window_seconds == 30 and self.sfreq == 100 and d_model == 80)
            or (
                self.input_window_seconds == 30 and self.sfreq == 125 and d_model == 100
            )
        ):
            warnings.warn(
                "This model was designed originally for input windows of 30sec at 100Hz, "
                "with d_model at 80 or at 125Hz, with d_model at 100, to use anything "
                "other than this may cause errors or cause the model to perform in "
                "other ways than intended",
                UserWarning,
            )

        # the usual kernel size for the mrcnn, for sfreq 100
        kernel_size = 7

        if self.sfreq == 125:
            kernel_size = 6

        mrcnn = _MRCNN(after_reduced_cnn_size, kernel_size)
        attn = _MultiHeadedAttention(n_attn_heads, d_model, after_reduced_cnn_size)
        ff = _PositionwiseFeedForward(d_model, d_ff, dropout)
        tce = _TCE(
            _EncoderLayer(
                d_model, deepcopy(attn), deepcopy(ff), after_reduced_cnn_size, dropout
            ),
            n_tce,
        )

        self.feature_extractor = nn.Sequential(mrcnn, tce)
        self.len_last_layer = self._len_last_layer(self.n_times)
        self.return_feats = return_feats

        # TODO: Add new way to handle return features
        """if return_feats:
            raise ValueError("return_feat == True is not accepted anymore")"""
        if not return_feats:
            self.final_layer = nn.Linear(
                d_model * after_reduced_cnn_size, self.n_outputs
            )

    def _len_last_layer(self, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(torch.Tensor(1, 1, input_size))
        self.feature_extractor.train()
        return len(out.flatten())

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        encoded_features = self.feature_extractor(x)
        encoded_features = encoded_features.contiguous().view(
            encoded_features.shape[0], -1
        )

        if self.return_feats:
            return encoded_features
        else:
            final_output = self.final_layer(encoded_features)
            return final_output

