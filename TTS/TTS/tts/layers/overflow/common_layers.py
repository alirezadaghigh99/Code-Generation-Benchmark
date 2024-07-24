class Encoder(nn.Module):
    r"""Neural HMM Encoder

    Same as Tacotron 2 encoder but increases the input length by states per phone

    Args:
        num_chars (int): Number of characters in the input.
        state_per_phone (int): Number of states per phone.
        in_out_channels (int): number of input and output channels.
        n_convolutions (int): number of convolutional layers.
    """

    def __init__(self, num_chars, state_per_phone, in_out_channels=512, n_convolutions=3):
        super().__init__()

        self.state_per_phone = state_per_phone
        self.in_out_channels = in_out_channels

        self.emb = nn.Embedding(num_chars, in_out_channels)
        self.convolutions = nn.ModuleList()
        for _ in range(n_convolutions):
            self.convolutions.append(ConvBNBlock(in_out_channels, in_out_channels, 5, "relu"))
        self.lstm = nn.LSTM(
            in_out_channels,
            int(in_out_channels / 2) * state_per_phone,
            num_layers=1,
            batch_first=True,
            bias=True,
            bidirectional=True,
        )
        self.rnn_state = None

    def forward(self, x: torch.FloatTensor, x_len: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Forward pass to the encoder.

        Args:
            x (torch.FloatTensor): input text indices.
                - shape: :math:`(b, T_{in})`
            x_len (torch.LongTensor): input text lengths.
                - shape: :math:`(b,)`

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: encoder outputs and output lengths.
                -shape: :math:`((b, T_{in} * states_per_phone, in_out_channels), (b,))`
        """
        b, T = x.shape
        o = self.emb(x).transpose(1, 2)
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        o = nn.utils.rnn.pack_padded_sequence(o, x_len.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        o = o.reshape(b, T * self.state_per_phone, self.in_out_channels)
        x_len = x_len * self.state_per_phone
        return o, x_len

    def inference(self, x, x_len):
        """Inference to the encoder.

        Args:
            x (torch.FloatTensor): input text indices.
                - shape: :math:`(b, T_{in})`
            x_len (torch.LongTensor): input text lengths.
                - shape: :math:`(b,)`

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: encoder outputs and output lengths.
                -shape: :math:`((b, T_{in} * states_per_phone, in_out_channels), (b,))`
        """
        b, T = x.shape
        o = self.emb(x).transpose(1, 2)
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        # self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        o = o.reshape(b, T * self.state_per_phone, self.in_out_channels)
        x_len = x_len * self.state_per_phone
        return o, x_len

class Outputnet(nn.Module):
    r"""
    This network takes current state and previous observed values as input
    and returns its parameters, mean, standard deviation and probability
    of transition to the next state
    """

    def __init__(
        self,
        encoder_dim: int,
        memory_rnn_dim: int,
        frame_channels: int,
        outputnet_size: List[int],
        flat_start_params: dict,
        std_floor: float = 1e-2,
    ):
        super().__init__()

        self.frame_channels = frame_channels
        self.flat_start_params = flat_start_params
        self.std_floor = std_floor

        input_size = memory_rnn_dim + encoder_dim
        output_size = 2 * frame_channels + 1

        self.parametermodel = ParameterModel(
            outputnet_size=outputnet_size,
            input_size=input_size,
            output_size=output_size,
            flat_start_params=flat_start_params,
            frame_channels=frame_channels,
        )

    def forward(self, ar_mels, inputs):
        r"""Inputs observation and returns the means, stds and transition probability for the current state

        Args:
            ar_mel_inputs (torch.FloatTensor): shape (batch, prenet_dim)
            states (torch.FloatTensor):  (batch, hidden_states, hidden_state_dim)

        Returns:
            means: means for the emission observation for each feature
                - shape: (B, hidden_states, feature_size)
            stds: standard deviations for the emission observation for each feature
                - shape: (batch, hidden_states, feature_size)
            transition_vectors: transition vector for the current hidden state
                - shape: (batch, hidden_states)
        """
        batch_size, prenet_dim = ar_mels.shape[0], ar_mels.shape[1]
        N = inputs.shape[1]

        ar_mels = ar_mels.unsqueeze(1).expand(batch_size, N, prenet_dim)
        ar_mels = torch.cat((ar_mels, inputs), dim=2)
        ar_mels = self.parametermodel(ar_mels)

        mean, std, transition_vector = (
            ar_mels[:, :, 0 : self.frame_channels],
            ar_mels[:, :, self.frame_channels : 2 * self.frame_channels],
            ar_mels[:, :, 2 * self.frame_channels :].squeeze(2),
        )
        std = F.softplus(std)
        std = self._floor_std(std)
        return mean, std, transition_vector

    def _floor_std(self, std):
        r"""
        It clamps the standard deviation to not to go below some level
        This removes the problem when the model tries to cheat for higher likelihoods by converting
        one of the gaussians to a point mass.

        Args:
            std (float Tensor): tensor containing the standard deviation to be
        """
        original_tensor = std.clone().detach()
        std = torch.clamp(std, min=self.std_floor)
        if torch.any(original_tensor != std):
            print(
                "[*] Standard deviation was floored! The model is preventing overfitting, nothing serious to worry about"
            )
        return std

