class EEGITNet(EEGModuleMixin, nn.Sequential):
    """EEG-ITNet: An Explainable Inception Temporal
     Convolutional Network for motor imagery classification from
     Salami et. al 2022.

    See [Salami2022]_ for details.

    Code adapted from https://github.com/abbassalami/eeg-itnet

    Parameters
    ----------
    drop_prob: float
        Dropout probability.
    n_classes: int
        Alias for n_outputs.
    in_channels: int
        Alias for n_chans.
    input_window_samples : int
        Alias for n_times.

    References
    ----------
    .. [Salami2022] A. Salami, J. Andreu-Perez and H. Gillmeister, "EEG-ITNet: An Explainable
    Inception Temporal Convolutional Network for motor imagery classification," in IEEE Access,
    doi: 10.1109/ACCESS.2022.3161489.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper based on author implementation.
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        n_times=None,
        drop_prob=0.4,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        n_classes=None,
        in_channels=None,
        input_window_samples=None,
        add_log_softmax=False,
    ):
        (
            n_outputs,
            n_chans,
            n_times,
        ) = deprecated_args(
            self,
            ("n_classes", "n_outputs", n_classes, n_outputs),
            ("in_channels", "n_chans", in_channels, n_chans),
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
            "classification.1.weight": "final_layer.clf.weight",
            "classification.1.bias": "final_layer.clf.weight",
        }

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_classes, in_channels, input_window_samples

        # ======== Handling EEG input ========================
        self.add_module(
            "input_preprocess",
            nn.Sequential(Ensure4d(), Rearrange("ba ch t 1 -> ba 1 ch t")),
        )
        # ======== Inception branches ========================
        block11 = self._get_inception_branch(
            in_channels=self.n_chans, out_channels=2, kernel_length=16
        )
        block12 = self._get_inception_branch(
            in_channels=self.n_chans, out_channels=4, kernel_length=32
        )
        block13 = self._get_inception_branch(
            in_channels=self.n_chans, out_channels=8, kernel_length=64
        )
        self.add_module("inception_block", _InceptionBlock((block11, block12, block13)))
        self.pool1 = self.add_module(
            "pooling",
            nn.Sequential(nn.AvgPool2d(kernel_size=(1, 4)), nn.Dropout(drop_prob)),
        )
        # =========== TC blocks =====================
        self.add_module(
            "TC_block1",
            _TCBlock(
                in_ch=14, kernel_length=4, dialation=1, padding=3, drop_prob=drop_prob
            ),
        )
        # ================================
        self.add_module(
            "TC_block2",
            _TCBlock(
                in_ch=14, kernel_length=4, dialation=2, padding=6, drop_prob=drop_prob
            ),
        )
        # ================================
        self.add_module(
            "TC_block3",
            _TCBlock(
                in_ch=14, kernel_length=4, dialation=4, padding=12, drop_prob=drop_prob
            ),
        )
        # ================================
        self.add_module(
            "TC_block4",
            _TCBlock(
                in_ch=14, kernel_length=4, dialation=8, padding=24, drop_prob=drop_prob
            ),
        )

        # ============= Dimensionality reduction ===================
        self.add_module(
            "dim_reduction",
            nn.Sequential(
                nn.Conv2d(14, 28, kernel_size=(1, 1)),
                nn.BatchNorm2d(28),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(drop_prob),
            ),
        )
        # ============== Classifier ==================
        # Moved flatten to another layer
        self.add_module("flatten", nn.Flatten())

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module(
            "clf", nn.Linear(int(int(self.n_times / 4) / 4) * 28, self.n_outputs)
        )

        if self.add_log_softmax:
            module.add_module("out_fun", nn.LogSoftmax(dim=1))
        else:
            module.add_module("out_fun", nn.Identity())

        self.add_module("final_layer", module)

    @staticmethod
    def _get_inception_branch(
        in_channels, out_channels, kernel_length, depth_multiplier=1
    ):
        return nn.Sequential(
            nn.Conv2d(
                1,
                out_channels,
                kernel_size=(1, kernel_length),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            _DepthwiseConv2d(
                out_channels,
                kernel_size=(in_channels, 1),
                depth_multiplier=depth_multiplier,
                bias=False,
                padding="valid",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

