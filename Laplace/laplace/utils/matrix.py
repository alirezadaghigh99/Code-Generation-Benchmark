def init_from_model(
        cls, model: nn.Module | Iterable[nn.Parameter], device: torch.device
    ) -> Kron:
        """Initialize Kronecker factors based on a models architecture.

        Parameters
        ----------
        model : nn.Module or iterable of parameters, e.g. model.parameters()
        device : torch.device

        Returns
        -------
        kron : Kron
        """
        if isinstance(model, torch.nn.Module):
            params = model.parameters()
        else:
            params = model

        kfacs = list()
        for p in params:
            if p.ndim == 1:  # bias
                P = p.size(0)
                kfacs.append([torch.zeros(P, P, device=device)])
            elif 4 >= p.ndim >= 2:  # fully connected or conv
                if p.ndim == 2:  # fully connected
                    P_in, P_out = p.size()
                else:
                    P_in, P_out = p.shape[0], np.prod(p.shape[1:])

                kfacs.append(
                    [
                        torch.zeros(P_in, P_in, device=device),
                        torch.zeros(P_out, P_out, device=device),
                    ]
                )
            else:
                raise ValueError("Invalid parameter shape in network.")
        return cls(kfacs)

