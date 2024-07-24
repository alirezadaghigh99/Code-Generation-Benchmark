class Stats:
    """A dict-compatible pytree containing the result of the statistics function."""

    mean: Union[float, complex] = _NaN
    """The mean value."""
    error_of_mean: float = _NaN
    """Estimate of the error of the mean."""
    variance: float = _NaN
    """Estimation of the variance of the data."""
    tau_corr: float = _NaN
    """Estimate of the autocorrelation time (in dimensionless units of number of steps).

    This value is estimated with a blocking algorithm by default, but the result is known
    to be unreliable. A more precise estimator based on the FFT transform can be used by
    setting the environment variable `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION=1`. This
    estimator is more computationally expensive, but overall the added cost should be
    negligible.
    """
    R_hat: float = _NaN
    """
    Estimator of the split-Rhat convergence estimator.

    The split-Rhat diagnostic is based on comparing intra-chain and inter-chain
    statistics of the sample and is thus only available for 2d-array inputs where
    the rows are independently sampled MCMC chains. In an ideal MCMC samples,
    R_hat should be 1.0. If it deviates from this value too much, this indicates
    MCMC convergence issues. Thresholds such as R_hat > 1.1 or even R_hat > 1.01 have
    been suggested in the literature for when to discard a sample. (See, e.g.,
    Gelman et al., `Bayesian Data Analysis <http://www.stat.columbia.edu/~gelman/book/>`_,
    or Vehtari et al., `arXiv:1903.08008 <https://arxiv.org/abs/1903.08008>`_.)
    """
    tau_corr_max: float = _NaN
    """
    Estimate of the maximum autocorrelation time among all Markov chains.

    This value is only computed if the environment variable
    `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION` is set.
    """

    def to_dict(self):
        jsd = {}
        jsd["Mean"] = _maybe_item(self.mean)
        jsd["Variance"] = _maybe_item(self.variance)
        jsd["Sigma"] = _maybe_item(self.error_of_mean)
        jsd["R_hat"] = _maybe_item(self.R_hat)
        jsd["TauCorr"] = _maybe_item(self.tau_corr)
        if config.netket_experimental_fft_autocorrelation:
            jsd["TauCorrMax"] = _maybe_item(self.tau_corr_max)
        return jsd

    def to_compound(self):
        return "Mean", self.to_dict()

    def __repr__(self):
        # extract adressable data from fully replicated arrays
        self = extract_replicated(self)
        mean, err, var = _format_decimal(self.mean, self.error_of_mean, self.variance)
        if not math.isnan(self.R_hat):
            ext = f", R̂={self.R_hat:.4f}"
        else:
            ext = ""
        if config.netket_experimental_fft_autocorrelation:
            if not (math.isnan(self.tau_corr) and math.isnan(self.tau_corr_max)):
                ext += f", τ={self.tau_corr:.1f}<{self.tau_corr_max:.1f}"
        return f"{mean} ± {err} [σ²={var}{ext}]"

    # Alias accessors
    def __getattr__(self, name):
        if name in ("mean", "Mean"):
            return self.mean
        elif name in ("variance", "Variance"):
            return self.variance
        elif name in ("error_of_mean", "Sigma"):
            return self.error_of_mean
        elif name in ("R_hat", "R"):
            return self.R_hat
        elif name in ("tau_corr", "TauCorr"):
            return self.tau_corr
        elif name in ("tau_corr_max", "TauCorrMax"):
            return self.tau_corr_max
        else:
            raise AttributeError(f"'Stats' object object has no attribute '{name}'")

    def real(self):
        return self.replace(mean=np.real(self.mean))

    def imag(self):
        return self.replace(mean=np.imag(self.mean))

