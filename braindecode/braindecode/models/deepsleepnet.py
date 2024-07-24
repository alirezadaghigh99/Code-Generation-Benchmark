class DeepSleepNet(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Supratak et al 2017.

    Convolutional neural network and bidirectional-Long Short-Term
    for single channels sleep staging described in [Supratak2017]_.

    Parameters
    ----------
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    n_classes :
        Alias for n_outputs.

    References
    ----------
    .. [Supratak2017] Supratak, A., Dong, H., Wu, C., & Guo, Y. (2017).
       DeepSleepNet: A model for automatic sleep stage scoring based
       on raw single-channel EEG. IEEE Transactions on Neural Systems
       and Rehabilitation Engineering, 25(11), 1998-2008.
    """

    def __init__(
        self,
        n_outputs=5,
        return_feats=False,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        n_classes=None,
    ):
        (n_outputs,) = deprecated_args(
            self,
            ("n_classes", "n_outputs", n_classes, n_outputs),
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
        del n_classes
        self.cnn1 = _SmallCNN()
        self.cnn2 = _LargeCNN()
        self.dropout = nn.Dropout(0.5)
        self.bilstm = _BiLSTM(input_size=3072, hidden_size=512, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(3072, 1024, bias=False), nn.BatchNorm1d(num_features=1024)
        )

        self.features_extractor = nn.Identity()
        self.len_last_layer = 1024
        self.return_feats = return_feats

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Linear(1024, self.n_outputs)
        else:
            self.final_layer = nn.Identity()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        if x.ndim == 3:
            x = x.unsqueeze(1)

        x1 = self.cnn1(x)
        x1 = x1.flatten(start_dim=1)

        x2 = self.cnn2(x)
        x2 = x2.flatten(start_dim=1)

        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        temp = x.clone()
        temp = self.fc(temp)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = x.squeeze()
        x = torch.add(x, temp)
        x = self.dropout(x)

        feats = self.features_extractor(x)

        if self.return_feats:
            return feats
        else:
            return self.final_layer(feats)

