class LSTMCellNNCF(RNNCellBaseNNCF):
    def __init__(self, input_size=1, hidden_size=1, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)
        self.cell = LSTMCellForwardNNCF(self.linear_list[0], self.linear_list[1])

    def forward(self, input_, hidden=None):
        self.check_forward_input(input_)
        if hidden is None:
            zeros = torch.zeros(input_.size(0), self.hidden_size, dtype=input_.dtype, device=input_.device)
            hidden = (zeros, zeros)
        self.check_forward_hidden(input_, hidden[0], "[0]")
        self.check_forward_hidden(input_, hidden[1], "[1]")

        return self.cell(input_, hidden)

class NNCF_RNN(nn.Module):
    """Common class for RNN modules. Currently, LSTM is supported only"""

    def __init__(
        self,
        mode="LSTM",
        input_size=1,
        hidden_size=1,
        num_layers=1,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        bias=True,
    ):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            nncf_logger.debug(
                f"dropout option adds dropout after all but last recurrent layer, "
                f"so non-zero dropout expects num_layers greater than 1, "
                f"but got dropout={dropout} and num_layers={num_layers}"
            )

        if mode == "LSTM":
            gate_size = 4 * hidden_size
            self.cell_type = LSTMCellForwardNNCF
        else:
            # elif mode == 'GRU':
            #     gate_size = 3 * hidden_size
            # elif mode == 'RNN_TANH':
            #     gate_size = hidden_size
            # elif mode == 'RNN_RELU':
            #     gate_size = hidden_size
            # else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._all_weights = []
        self.cells = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                linear_ih = nn.Linear(layer_input_size, gate_size, bias)
                linear_hh = nn.Linear(hidden_size, gate_size, bias)
                self.cells.append(self.cell_type(linear_ih, linear_hh))
                params = (linear_ih.weight, linear_hh.weight, linear_ih.bias, linear_hh.bias)
                suffix = "_reverse" if direction == 1 else ""
                weight_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    weight_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                weight_names = [x.format(layer, suffix) for x in weight_names]
                for name, param in zip(weight_names, params):
                    setattr(self, name, param)
                self._all_weights.append(weight_names)

        self.reset_parameters()
        self.variable_length = True
        self.rnn_impl = self.get_rnn_impl(self.variable_length, self.cells)

    def get_rnn_impl(self, variable_length, cells):
        if variable_length:
            rec_factory = variable_recurrent_factory()
        else:
            rec_factory = Recurrent
        inners = []
        for layer_idx in range(self.num_layers):
            idx = layer_idx * self.num_directions
            if self.bidirectional:
                layer_inners = [rec_factory(cells[idx]), rec_factory(cells[idx + 1], reverse=True)]
            else:
                layer_inners = [
                    rec_factory(cells[idx]),
                ]
            inners.extend(layer_inners)
        return StackedRNN(inners, self.num_layers, (self.mode == "LSTM"), dropout=self.dropout)

    def check_forward_args(self, input_, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input_.dim() != expected_input_dim:
            raise nncf.ValidationError(
                "input_ must have {} dimensions, got {}".format(expected_input_dim, input_.dim())
            )
        if self.input_size != input_.size(-1):
            raise nncf.ValidationError(
                "input_.size(-1) must be equal to input_size. Expected {}, got {}".format(
                    self.input_size, input_.size(-1)
                )
            )

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input_.size(0) if self.batch_first else input_.size(1)

        expected_hidden_size = (mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg="Expected hidden size {}, got {}"):
            expected_size = self.num_layers * self.num_directions
            if expected_size != len(hx):
                raise nncf.InternalError("Expected number of hidden states {}, got {}".format(expected_size, len(hx)))
            for element in hx:
                if tuple(element.size()) != expected_hidden_size:
                    raise nncf.InternalError(msg.format(expected_hidden_size, tuple(element.size())))

        if self.mode == "LSTM":
            check_hidden_size(hidden[0], expected_hidden_size, "Expected hidden[0] size {}, got {}")
            check_hidden_size(hidden[1], expected_hidden_size, "Expected hidden[1] size {}, got {}")
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    @staticmethod
    def apply_permutation(tensor: torch.Tensor, permutation: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return tensor.index_select(dim, permutation)

    def permute_hidden(
        self, hx: torch.Tensor, permutation: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if permutation is None:
            return hx
        return self.apply_permutation(hx[0], permutation), self.apply_permutation(hx[1], permutation)

    def prepare_hidden(
        self, hx: torch.Tensor, permutation: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]]:
        if permutation is None:
            return hx
        split_size = len(hx[0])
        concat_hx = torch.cat([torch.unsqueeze(t, 0) for t in hx[0]])
        concat_cx = torch.cat([torch.unsqueeze(t, 0) for t in hx[1]])
        permuted_hidden = self.apply_permutation(concat_hx, permutation), self.apply_permutation(concat_cx, permutation)
        hc = permuted_hidden[0].chunk(split_size, 0)
        cc = permuted_hidden[1].chunk(split_size, 0)
        hidden = (tuple(torch.squeeze(c, 0) for c in hc), tuple(torch.squeeze(c, 0) for c in cc))
        return hidden

    def forward(self, input_, hidden=None):
        is_packed = isinstance(input_, PackedSequence)

        sorted_indices = None
        unsorted_indices = None
        if is_packed:
            input_, batch_sizes, sorted_indices, unsorted_indices = input_
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input_.size(0) if self.batch_first else input_.size(1)

        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            hidden = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                requires_grad=False,
                device=input_.device,
            )
            if self.mode == "LSTM":
                hidden = (hidden, hidden)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hidden = self.prepare_hidden(hidden, sorted_indices)

        self.check_forward_args(input_, hidden, batch_sizes)

        is_currently_variable = batch_sizes is not None
        if self.variable_length and not is_currently_variable or not self.variable_length and is_currently_variable:
            # override rnn_impl, it's assumed that this should happen very seldom, as
            # usually there's only one mode active whether variable length, or constant ones
            self.rnn_impl = self.get_rnn_impl(is_currently_variable, self.cells)

        if self.batch_first and batch_sizes is None:
            input_ = input_.transpose(0, 1)

        hidden, output = self.rnn_impl(input_, hidden, batch_sizes)

        if self.batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)

        return output, self.permute_hidden(hidden, unsorted_indices)

