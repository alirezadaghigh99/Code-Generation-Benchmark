class FullLaplace(ParametricLaplace):
    """Laplace approximation with full, i.e., dense, log likelihood Hessian approximation
    and hence posterior precision. Based on the chosen `backend` parameter, the full
    approximation can be, for example, a generalized Gauss-Newton matrix.
    Mathematically, we have \\(P \\in \\mathbb{R}^{P \\times P}\\).
    See `BaseLaplace` for the full interface.
    """

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ("all", "full")

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            model,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
            enable_backprop,
            dict_key_x,
            dict_key_y,
            backend,
            backend_kwargs,
        )
        self._posterior_scale: torch.Tensor | None = None

    def _init_H(self) -> None:
        self.H: torch.Tensor = torch.zeros(
            self.n_params, self.n_params, device=self._device
        )

    def _curv_closure(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.full(X, y, N=N)

    def fit(
        self,
        train_loader: DataLoader,
        override: bool = True,
        progress_bar: bool = False,
    ) -> None:
        self._posterior_scale = None
        super().fit(train_loader, override=override, progress_bar=progress_bar)

    def _compute_scale(self) -> None:
        self._posterior_scale = invsqrt_precision(self.posterior_precision)

    @property
    def posterior_scale(self) -> torch.Tensor:
        """Posterior scale (square root of the covariance), i.e.,
        \\(P^{-\\frac{1}{2}}\\).

        Returns
        -------
        scale : torch.tensor
            `(parameters, parameters)`
        """
        if self._posterior_scale is None:
            self._compute_scale()
        return self._posterior_scale

    @property
    def posterior_covariance(self) -> torch.Tensor:
        """Posterior covariance, i.e., \\(P^{-1}\\).

        Returns
        -------
        covariance : torch.tensor
            `(parameters, parameters)`
        """
        scale = self.posterior_scale
        return scale @ scale.T

    @property
    def posterior_precision(self) -> torch.Tensor:
        """Posterior precision \\(P\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters, parameters)`
        """
        self._check_H_init()
        return self._H_factor * self.H + torch.diag(self.prior_precision_diag)

    @property
    def log_det_posterior_precision(self) -> torch.Tensor:
        return self.posterior_precision.logdet()

    def square_norm(self, value: torch.Tensor) -> torch.Tensor:
        delta = value - self.mean
        return delta @ self.posterior_precision @ delta

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncp,pq,nkq->nck", Js, self.posterior_covariance, Js)

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        n_batch, n_outs, n_params = Js.shape
        Js = Js.reshape(n_batch * n_outs, n_params)
        return torch.einsum("np,pq,mq->nm", Js, self.posterior_covariance, Js)

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples, self.n_params, device=self._device, generator=generator
        )
        # (n_samples, n_params) x (n_params, n_params) -> (n_samples, n_params)
        samples = samples @ self.posterior_scale
        return self.mean.reshape(1, self.n_params) + samples

class DiagLaplace(ParametricLaplace):
    """Laplace approximation with diagonal log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have \\(P \\approx \\textrm{diag}(P)\\).
    See `BaseLaplace` for the full interface.
    """

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ("all", "diag")

    def _init_H(self) -> None:
        self.H: torch.Tensor = torch.zeros(self.n_params, device=self._device)

    def _curv_closure(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.diag(X, y, N=N, **self._asdl_fisher_kwargs)

    @property
    def posterior_precision(self) -> torch.Tensor:
        """Diagonal posterior precision \\(p\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        self._check_H_init()
        return self._H_factor * self.H + self.prior_precision_diag

    @property
    def posterior_scale(self) -> torch.Tensor:
        """Diagonal posterior scale \\(\\sqrt{p^{-1}}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return 1 / self.posterior_precision.sqrt()

    @property
    def posterior_variance(self) -> torch.Tensor:
        """Diagonal posterior variance \\(p^{-1}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return 1 / self.posterior_precision

    @property
    def log_det_posterior_precision(self) -> torch.Tensor:
        return self.posterior_precision.log().sum()

    def square_norm(self, value: torch.Tensor) -> torch.Tensor:
        delta = value - self.mean
        return delta @ (delta * self.posterior_precision)

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        return torch.einsum("ncp,p,nkp->nck", Js, self.posterior_variance, Js)

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        n_batch, n_outs, n_params = Js.shape
        Js = Js.reshape(n_batch * n_outs, n_params)
        cov = torch.einsum("np,p,mp->nm", Js, self.posterior_variance, Js)
        return cov

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples, self.n_params, device=self._device, generator=generator
        )
        samples = samples * self.posterior_scale.reshape(1, self.n_params)
        return self.mean.reshape(1, self.n_params) + samples

