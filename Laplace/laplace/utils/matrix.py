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

class Kron:
    """Kronecker factored approximate curvature representation for a corresponding
    neural network.
    Each element in `kfacs` is either a tuple or single matrix.
    A tuple represents two Kronecker factors \\(Q\\), and \\(H\\) and a single element
    is just a full block Hessian approximation.

    Parameters
    ----------
    kfacs : list[Iterable[torch.Tensor] | torch.Tensor]
        each element in the list is a tuple of two Kronecker factors Q, H
        or a single matrix approximating the Hessian (in case of bias, for example)
    """

    def __init__(self, kfacs: list[tuple[torch.Tensor] | torch.Tensor]) -> None:
        self.kfacs: list[tuple[torch.Tensor] | torch.Tensor] = kfacs

    @classmethod
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

    def __add__(self, other: Kron) -> Kron:
        """Add up Kronecker factors `self` and `other`.

        Parameters
        ----------
        other : Kron

        Returns
        -------
        kron : Kron
        """
        if not isinstance(other, Kron):
            raise ValueError("Can only add Kron to Kron.")

        kfacs = [
            [Hi.add(Hj) for Hi, Hj in zip(Fi, Fj)]
            for Fi, Fj in zip(self.kfacs, other.kfacs)
        ]

        return Kron(kfacs)

    def __mul__(self, scalar: float | torch.Tensor) -> Kron:
        """Multiply all Kronecker factors by scalar.
        The multiplication is distributed across the number of factors
        using `pow(scalar, 1 / len(F))`. `len(F)` is either `1` or `2`.

        Parameters
        ----------
        scalar : float, torch.Tensor

        Returns
        -------
        kron : Kron
        """
        if not _is_valid_scalar(scalar):
            raise ValueError("Input not valid python or torch scalar.")

        # distribute factors evenly so that each group is multiplied by factor
        kfacs = [[pow(scalar, 1 / len(F)) * Hi for Hi in F] for F in self.kfacs]
        return Kron(kfacs)

    def __len__(self) -> int:
        return len(self.kfacs)

    def decompose(self, damping: bool = False) -> KronDecomposed:
        """Eigendecompose Kronecker factors and turn into `KronDecomposed`.
        Parameters
        ----------
        damping : bool
            use damping

        Returns
        -------
        kron_decomposed : KronDecomposed
        """
        eigvecs, eigvals = list(), list()
        for F in self.kfacs:
            Qs, ls = list(), list()
            for Hi in F:
                if Hi.ndim > 1:
                    # Dense Kronecker factor.
                    eigval, Q = symeig(Hi)
                else:
                    # Diagonal Kronecker factor.
                    eigval = Hi
                    # This might be too memory intensive since len(Hi) can be large.
                    Q = torch.eye(len(Hi), dtype=Hi.dtype, device=Hi.device)
                Qs.append(Q)
                ls.append(eigval)
            eigvecs.append(Qs)
            eigvals.append(ls)
        return KronDecomposed(eigvecs, eigvals, damping=damping)

    def _bmm(self, W: torch.Tensor) -> torch.Tensor:
        """Implementation of `bmm` which casts the parameters to the right shape.

        Parameters
        ----------
        W : torch.Tensor
            matrix `(batch, classes, params)`

        Returns
        -------
        SW : torch.Tensor
            result `(batch, classes, params)`
        """
        # self @ W[batch, k, params]
        assert len(W.size()) == 3
        B, K, P = W.size()
        W = W.reshape(B * K, P)
        cur_p = 0
        SW = list()
        for Fs in self.kfacs:
            if len(Fs) == 1:
                Q = Fs[0]
                p = len(Q)
                W_p = W[:, cur_p : cur_p + p].T
                SW.append((Q @ W_p).T if Q.ndim > 1 else (Q.view(-1, 1) * W_p).T)
                cur_p += p
            elif len(Fs) == 2:
                Q, H = Fs
                p_in, p_out = len(Q), len(H)
                p = p_in * p_out
                W_p = W[:, cur_p : cur_p + p].reshape(B * K, p_in, p_out)
                QW_p = Q @ W_p if Q.ndim > 1 else Q.view(-1, 1) * W_p
                QW_pHt = QW_p @ H.T if H.ndim > 1 else QW_p * H.view(1, -1)
                SW.append(QW_pHt.reshape(B * K, p_in * p_out))
                cur_p += p
            else:
                raise AttributeError("Shape mismatch")
        SW = torch.cat(SW, dim=1).reshape(B, K, P)
        return SW

    def bmm(self, W: torch.Tensor, exponent: float = 1) -> torch.Tensor:
        """Batched matrix multiplication with the Kronecker factors.
        If Kron is `H`, we compute `H @ W`.
        This is useful for computing the predictive or a regularization
        based on Kronecker factors as in continual learning.

        Parameters
        ----------
        W : torch.Tensor
            matrix `(batch, classes, params)`
        exponent: float, default=1
            only can be `1` for Kron, requires `KronDecomposed` for other
            exponent values of the Kronecker factors.

        Returns
        -------
        SW : torch.Tensor
            result `(batch, classes, params)`
        """
        if exponent != 1:
            raise ValueError("Only supported after decomposition.")
        if W.ndim == 1:
            return self._bmm(W.unsqueeze(0).unsqueeze(0)).squeeze()
        elif W.ndim == 2:
            return self._bmm(W.unsqueeze(1)).squeeze()
        elif W.ndim == 3:
            return self._bmm(W)
        else:
            raise ValueError("Invalid shape for W")

    def logdet(self) -> torch.Tensor:
        """Compute log determinant of the Kronecker factors and sums them up.
        This corresponds to the log determinant of the entire Hessian approximation.

        Returns
        -------
        logdet : torch.Tensor
        """
        logdet = 0
        for F in self.kfacs:
            if len(F) == 1:
                logdet += F[0].logdet() if F[0].ndim > 1 else F[0].log().sum()
            else:  # len(F) == 2
                Hi, Hj = F
                p_in, p_out = len(Hi), len(Hj)
                logdet += p_out * Hi.logdet() if Hi.ndim > 1 else p_out * Hi.log().sum()
                logdet += p_in * Hj.logdet() if Hj.ndim > 1 else p_in * Hj.log().sum()
        return logdet

    def diag(self) -> torch.Tensor:
        """Extract diagonal of the entire Kronecker factorization.

        Returns
        -------
        diag : torch.Tensor
        """
        diags = list()
        for F in self.kfacs:
            F0 = F[0].diag() if F[0].ndim > 1 else F[0]
            if len(F) == 1:
                diags.append(F0)
            else:
                F1 = F[1].diag() if F[1].ndim > 1 else F[1]
                diags.append(torch.outer(F0, F1).flatten())
        return torch.cat(diags)

    def to_matrix(self) -> torch.Tensor:
        """Make the Kronecker factorization dense by computing the kronecker product.
        Warning: this should only be used for testing purposes as it will allocate
        large amounts of memory for big architectures.

        Returns
        -------
        block_diag : torch.Tensor
        """
        blocks = list()
        for F in self.kfacs:
            F0 = F[0] if F[0].ndim > 1 else F[0].diag()
            if len(F) == 1:
                blocks.append(F0)
            else:
                F1 = F[1] if F[1].ndim > 1 else F[1].diag()
                blocks.append(kron(F0, F1))
        return block_diag(blocks)

    # for commutative operations
    __radd__ = __add__
    __rmul__ = __mul__

