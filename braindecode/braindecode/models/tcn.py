class TCN(EEGModuleMixin, nn.Module):
    """Temporal Convolutional Network (TCN) from Bai et al 2018.

    See [Bai2018]_ for details.

    Code adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

    Parameters
    ----------
    n_filters: int
        number of output filters of each convolution
    n_blocks: int
        number of temporal blocks in the network
    kernel_size: int
        kernel size of the convolutions
    drop_prob: float
        dropout probability
    n_in_chans: int
        Alias for `n_chans`.

    References
    ----------
    .. [Bai2018] Bai, S., Kolter, J. Z., & Koltun, V. (2018).
       An empirical evaluation of generic convolutional and recurrent networks
       for sequence modeling.
       arXiv preprint arXiv:1803.01271.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_blocks=3,
        n_filters=30,
        kernel_size=5,
        drop_prob=0.5,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        n_in_chans=None,
        add_log_softmax=False,
    ):
        (n_chans,) = deprecated_args(
            self,
            ("n_in_chans", "n_chans", n_in_chans, n_chans),
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
        del n_in_chans

        self.mapping = {
            "fc.weight": "final_layer.fc.weight",
            "fc.bias": "final_layer.fc.bias",
        }
        self.ensuredims = Ensure4d()
        t_blocks = nn.Sequential()
        for i in range(n_blocks):
            n_inputs = self.n_chans if i == 0 else n_filters
            dilation_size = 2**i
            t_blocks.add_module(
                "temporal_block_{:d}".format(i),
                TemporalBlock(
                    n_inputs=n_inputs,
                    n_outputs=n_filters,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    drop_prob=drop_prob,
                ),
            )
        self.temporal_blocks = t_blocks

        # Here, change to final_layer
        self.final_layer = _FinalLayer(
            in_features=n_filters,
            out_features=self.n_outputs,
            add_log_softmax=add_log_softmax,
        )
        self.min_len = 1
        for i in range(n_blocks):
            dilation = 2**i
            self.min_len += 2 * (kernel_size - 1) * dilation

        # start in eval mode
        self.eval()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        x = self.ensuredims(x)
        # x is in format: B x C x T x 1
        (batch_size, _, time_size, _) = x.size()
        assert time_size >= self.min_len
        # remove empty trailing dimension
        x = x.squeeze(3)
        x = self.temporal_blocks(x)
        # Convert to: B x T x C
        x = x.transpose(1, 2).contiguous()

        out = self.final_layer(x, batch_size, time_size, self.min_len)

        return out

