class SPARCNet(EEGModuleMixin, nn.Module):
    """Seizures, Periodic and Rhythmic pattern Continuum Neural Network (SPaRCNet) [jing2023]_.

    This is a temporal CNN model for biosignal classification based on the DenseNet
    architecture.

    The model is based on the unofficial implementation [Code2023]_.

    .. versionadded:: 0.9

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors.

    Parameters
    ----------
    block_layers : int, optional
        Number of layers per dense block. Default is 4.
    growth_rate : int, optional
        Growth rate of the DenseNet. Default is 16.
    bn_size : int, optional
        Bottleneck size. Default is 16.
    drop_rate : float, optional
        Dropout rate. Default is 0.5.
    conv_bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.
    batch_norm : bool, optional
        Whether to use batch normalization. Default is True.

    References
    ----------
    .. [jing2023] Jing, J., Ge, W., Hong, S., Fernandes, M. B., Lin, Z.,
       Yang, C., ... & Westover, M. B. (2023). Development of expert-level
       classification of seizures and rhythmic and periodic
       patterns during eeg interpretation. Neurology, 100(17), e1750-e1762.
    .. [Code2023] Yang, C., Westover, M.B. and Sun, J., 2023. BIOT
       Biosignal Transformer for Cross-data Learning in the Wild.
       GitHub https://github.com/ycq091044/BIOT (accessed 2024-02-13)

    """

    def __init__(
        self,
        n_chans: int | None = None,
        n_times: int | None = None,
        n_outputs: int | None = None,
        # Neural network parameters
        block_layers: int = 4,
        growth_rate: int = 16,
        bottleneck_size: int = 16,
        drop_rate: float = 0.5,
        conv_bias: bool = True,
        batch_norm: bool = True,
        # EEGModuleMixin parameters
        # (another way to present the same parameters)
        chs_info: list[dict[Any, Any]] | None = None,
        input_window_seconds: float | None = None,
        sfreq: int | None = None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds

        # add initial convolutional layer
        # the number of output channels is the smallest power of 2
        # that is greater than the number of input channels
        out_channels = 2 ** (floor(log2(self.n_chans)) + 1)
        first_conv = OrderedDict(
            [
                (
                    "conv0",
                    nn.Conv1d(
                        in_channels=self.n_chans,
                        out_channels=out_channels,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=conv_bias,
                    ),
                )
            ]
        )
        first_conv["norm0"] = nn.BatchNorm1d(out_channels)
        first_conv["act_layer"] = nn.ELU()
        first_conv["pool0"] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.encoder = nn.Sequential(first_conv)

        n_channels = out_channels

        # Adding dense blocks
        for n_layer in range(floor(log2(self.n_times // 4))):
            block = DenseBlock(
                num_layers=block_layers,
                in_channels=n_channels,
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                drop_rate=drop_rate,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("denseblock%d" % (n_layer + 1), block)
            # update the number of channels after each dense block
            n_channels = n_channels + block_layers * growth_rate

            trans = TransitionLayer(
                in_channels=n_channels,
                out_channels=n_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("transition%d" % (n_layer + 1), trans)
            # update the number of channels after each transition layer
            n_channels = n_channels // 2

        # add final convolutional layer
        self.final_layer = nn.Sequential(
            nn.ELU(),
            nn.Linear(n_channels, self.n_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the model.

        Official init from torch repo, using kaiming_normal for conv layers
        and normal for linear layers.

        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, X: torch.Tensor):
        """
        Forward pass of the model.

        Parameters
        ----------
        X: torch.Tensor
            The input tensor of the model with shape (batch_size, n_channels, n_times)

        Returns
        -------
        torch.Tensor
            The output tensor of the model with shape (batch_size, n_outputs)
        """
        emb = self.encoder(X).squeeze(-1)
        out = self.final_layer(emb)
        return out

