def fit(
        self,
        train_loader: DataLoader,
        override: bool = True,
        progress_bar: bool = False,
    ) -> None:
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
        override : bool, default=True
            whether to initialize H, loss, and n_data again; setting to False is useful for
            online learning settings to accumulate a sequential posterior approximation.
        progress_bar: bool, default=False
        """
        if not override:
            raise ValueError(
                "Last-layer Laplace approximations do not support `override=False`."
            )

        self.model.eval()

        if self.model.last_layer is None:
            self.data: tuple[torch.Tensor, torch.Tensor] | MutableMapping = next(
                iter(train_loader)
            )
            self._find_last_layer(self.data)
            params: torch.Tensor = parameters_to_vector(
                self.model.last_layer.parameters()
            ).detach()
            self.n_params: int = len(params)
            self.n_layers: int = len(list(self.model.last_layer.parameters()))
            # here, check the already set prior precision again
            self.prior_precision: float | torch.Tensor = self._prior_precision
            self.prior_mean: float | torch.Tensor = self._prior_mean
            self._init_H()

        super().fit(train_loader, override=override)
        self.mean: torch.Tensor = parameters_to_vector(
            self.model.last_layer.parameters()
        )

        if not self.enable_backprop:
            self.mean = self.mean.detach()

