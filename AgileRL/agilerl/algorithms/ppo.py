    def prepare_state(self, state):
        """Prepares state for forward pass through neural network.

        :param state: Observation of environment
        :type state: np.Array() or list
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()

        if self.accelerator is None:
            state = state.to(self.device)
        else:
            state = state.to(self.accelerator.device)

        if self.one_hot:
            state = (
                nn.functional.one_hot(state.long(), num_classes=self.state_dim[0])
                .float()
                .squeeze()
            )

        if (self.arch == "mlp" and len(state.size()) < 2) or (
            self.arch == "cnn" and len(state.size()) < 4
        ):
            state = state.unsqueeze(0)

        return state.float()