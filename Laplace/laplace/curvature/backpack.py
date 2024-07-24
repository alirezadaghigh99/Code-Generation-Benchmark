class BackPackGGN(BackPackInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Backpack."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        stochastic: bool = False,
    ):
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )
        self.stochastic = stochastic

    def _get_diag_ggn(self) -> torch.Tensor:
        if self.stochastic:
            return torch.cat(
                [p.diag_ggn_mc.data.flatten() for p in self._model.parameters()]
            )
        else:
            return torch.cat(
                [p.diag_ggn_exact.data.flatten() for p in self._model.parameters()]
            )

    def _get_kron_factors(self) -> Kron:
        if self.stochastic:
            return Kron([p.kfac for p in self._model.parameters()])
        else:
            return Kron([p.kflr for p in self._model.parameters()])

    @staticmethod
    def _rescale_kron_factors(kron: Kron, M: int, N: int) -> Kron:
        # Renormalize Kronecker factor to sum up correctly over N data points with batches of M
        # for M=N (full-batch) just M/N=1
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= M / N
        return kron

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = DiagGGNMC if self.stochastic else DiagGGNExact
        f = self.model(x)
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.likelihood == "regression" else f.view(-1, f.size(-1))
        y = y if self.likelihood == "regression" else y.view(-1)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        dggn = self._get_diag_ggn()
        if self.subnetwork_indices is not None:
            dggn = dggn[self.subnetwork_indices]

        return self.factor * loss.detach(), self.factor * dggn

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        context = KFAC if self.stochastic else KFLR
        f = self.model(x)
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.likelihood == "regression" else f.view(-1, f.size(-1))
        y = y if self.likelihood == "regression" else y.view(-1)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        kron = self._get_kron_factors()
        kron = self._rescale_kron_factors(kron, len(y), N)

        return self.factor * loss.detach(), self.factor * kron

