class EEGResNet(EEGModuleMixin, nn.Sequential):
    """Residual Network for EEG  from Schirrmeister et al 2017.

    Model described in [Schirrmeister2017]_.

    Parameters
    ----------
     in_chans :
        Alias for `n_chans`.
     n_classes :
        Alias for `n_outputs`.
     input_window_samples :
        Alias for `n_times`.

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        final_pool_length="auto",
        n_first_filters=20,
        n_layers_per_block=2,
        first_filter_length=3,
        nonlinearity=elu,
        split_first_layer=True,
        batch_norm_alpha=0.1,
        batch_norm_epsilon=1e-4,
        conv_weight_init_fn=lambda w: init.kaiming_normal_(w, a=0),
        chs_info=None,
        input_window_seconds=None,
        sfreq=250,
        in_chans=None,
        n_classes=None,
        input_window_samples=None,
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
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples

        if final_pool_length == "auto":
            assert self.n_times is not None
        assert first_filter_length % 2 == 1
        self.final_pool_length = final_pool_length
        self.n_first_filters = n_first_filters
        self.n_layers_per_block = n_layers_per_block
        self.first_filter_length = first_filter_length
        self.nonlinearity = nonlinearity
        self.split_first_layer = split_first_layer
        self.batch_norm_alpha = batch_norm_alpha
        self.batch_norm_epsilon = batch_norm_epsilon
        self.conv_weight_init_fn = conv_weight_init_fn

        self.mapping = {
            "conv_classifier.weight": "final_layer.conv_classifier.weight",
            "conv_classifier.bias": "final_layer.conv_classifier.bias",
        }

        self.add_module("ensuredims", Ensure4d())
        if self.split_first_layer:
            self.add_module("dimshuffle", Rearrange("batch C T 1 -> batch 1 T C"))
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_first_filters,
                    (self.first_filter_length, 1),
                    stride=1,
                    padding=(self.first_filter_length // 2, 0),
                ),
            )
            self.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_first_filters,
                    self.n_first_filters,
                    (1, self.n_chans),
                    stride=(1, 1),
                    bias=False,
                ),
            )
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.n_chans,
                    self.n_first_filters,
                    (self.first_filter_length, 1),
                    stride=(1, 1),
                    padding=(self.first_filter_length // 2, 0),
                    bias=False,
                ),
            )
        n_filters_conv = self.n_first_filters
        self.add_module(
            "bnorm",
            nn.BatchNorm2d(
                n_filters_conv, momentum=self.batch_norm_alpha, affine=True, eps=1e-5
            ),
        )
        self.add_module("conv_nonlin", Expression(self.nonlinearity))
        cur_dilation = np.array([1, 1])
        n_cur_filters = n_filters_conv
        i_block = 1
        for i_layer in range(self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )
        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(2 * n_cur_filters)
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_out_filters,
                dilation=cur_dilation,
            ),
        )
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(1.5 * n_cur_filters)
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_out_filters,
                dilation=cur_dilation,
            ),
        )
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_cur_filters,
                dilation=cur_dilation,
            ),
        )
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_cur_filters,
                dilation=cur_dilation,
            ),
        )
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_cur_filters,
                dilation=cur_dilation,
            ),
        )
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )
        i_block += 1
        cur_dilation[0] *= 2
        self.add_module(
            "res_{:d}_{:d}".format(i_block, 0),
            _ResidualBlock(
                n_cur_filters,
                n_cur_filters,
                dilation=cur_dilation,
            ),
        )
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module(
                "res_{:d}_{:d}".format(i_block, i_layer),
                _ResidualBlock(n_cur_filters, n_cur_filters, dilation=cur_dilation),
            )

        self.eval()
        if self.final_pool_length == "auto":
            self.add_module("mean_pool", nn.AdaptiveAvgPool2d((1, 1)))
        else:
            pool_dilation = int(cur_dilation[0]), int(cur_dilation[1])
            self.add_module(
                "mean_pool",
                AvgPool2dWithConv(
                    (self.final_pool_length, 1), (1, 1), dilation=pool_dilation
                ),
            )

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()

        module.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_cur_filters,
                self.n_outputs,
                (1, 1),
                bias=True,
            ),
        )

        if self.add_log_softmax:
            module.add_module("logsoftmax", nn.LogSoftmax(dim=1))

        module.add_module("squeeze", Expression(squeeze_final_output))

        self.add_module("final_layer", module)

        # Initialize all weights
        self.apply(lambda module: _weights_init(module, self.conv_weight_init_fn))

        # Start in eval mode
        self.eval()

