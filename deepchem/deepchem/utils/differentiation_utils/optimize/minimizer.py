class TerminationCondition(object):
    """The class to handle the stopping conditions.

    Examples
    --------
    >>> stop_cond = TerminationCondition(1e-8, 1e-8, 1e-8, 1e-8, True)

    """

    def __init__(self, f_tol: float, f_rtol: float, x_tol: float, x_rtol: float,
                 verbose: bool):
        """Initialize the TerminationCondition.

        Parameters
        ----------
        f_tol: float or None
            Absolute tolerance of the output ``f``.
        f_rtol: float or None
            Relative tolerance of the output ``f``.
        x_tol: float or None
            Absolute tolerance of the norm of the input ``x``.
        x_rtol: float or None
            Relative tolerance of the norm of the input ``x``.
        verbose: bool
            Whether to print the iteration information.

        """
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.verbose = verbose

        self._ever_converge = False
        self._max_i = -1
        self._best_dxnorm = float("inf")
        self._best_df = float("inf")
        self._best_f = float("inf")
        self._best_x: Optional[torch.Tensor] = None

    def to_stop(self, i: int, xnext: torch.Tensor, x: torch.Tensor,
                f: torch.Tensor, fprev: torch.Tensor) -> bool:
        """Check if the stopping conditions are met.

        Parameters
        ----------
        i: int
            The iteration number.
        xnext: torch.Tensor
            The next input.
        x: torch.Tensor
            The current input.
        f: torch.Tensor
            The current output.
        fprev: torch.Tensor
            The previous output.

        Returns
        -------
        bool
            Whether to stop the iteration.

        """
        xnorm: float = float(x.detach().norm().item())
        dxnorm: float = float((x - xnext).detach().norm().item())
        fabs: float = float(f.detach().abs().item())
        df: float = float((fprev - f).detach().abs().item())
        fval: float = float(f.detach().item())

        xtcheck = dxnorm < self.x_tol
        xrcheck = dxnorm < self.x_rtol * xnorm
        ytcheck = df < self.f_tol
        yrcheck = df < self.f_rtol * fabs
        converge = xtcheck or xrcheck or ytcheck or yrcheck
        if self.verbose:
            if i == 0:
                print("   #:             f |        dx,        df")
            if converge:
                print("Finish with convergence")
            if i == 0 or ((i + 1) % 10) == 0 or converge:
                print("%4d: %.6e | %.3e, %.3e" % (i + 1, f, dxnorm, df))

        res = (i > 0 and converge)

        # get the best values
        if not self._ever_converge and res:
            self._ever_converge = True
        if i > self._max_i:
            self._max_i = i
        if fval < self._best_f:
            self._best_f = fval
            self._best_x = x
            self._best_dxnorm = dxnorm
            self._best_df = df
        return res

    def get_best_x(self, x: torch.Tensor) -> torch.Tensor:
        """Get the best input.

        Parameters
        ----------
        x: torch.Tensor
            The current input.

        Returns
        -------
        torch.Tensor
            The best input.

        """
        # usually user set maxiter == 0 just to wrap the minimizer backprop
        if not self._ever_converge and self._max_i > -1:
            msg = (
                "The minimizer does not converge after %d iterations. "
                "Best |dx|=%.4e, |df|=%.4e, f=%.4e" %
                (self._max_i, self._best_dxnorm, self._best_df, self._best_f))
            warnings.warn(msg)
            assert isinstance(self._best_x, torch.Tensor)
            return self._best_x
        else:
            return x

