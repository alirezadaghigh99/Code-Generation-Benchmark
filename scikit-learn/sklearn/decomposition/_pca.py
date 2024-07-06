def _assess_dimension(spectrum, rank, n_samples):
    """Compute the log-likelihood of a rank ``rank`` dataset.

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``. This implements the method of
    T. P. Minka.

    Parameters
    ----------
    spectrum : ndarray of shape (n_features,)
        Data spectrum.
    rank : int
        Tested rank value. It should be strictly lower than n_features,
        otherwise the method isn't specified (division by zero in equation
        (31) from the paper).
    n_samples : int
        Number of samples.

    Returns
    -------
    ll : float
        The log-likelihood.

    References
    ----------
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
    <https://proceedings.neurips.cc/paper/2000/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf>`_
    """
    xp, _ = get_namespace(spectrum)

    n_features = spectrum.shape[0]
    if not 1 <= rank < n_features:
        raise ValueError("the tested rank should be in [1, n_features - 1]")

    eps = 1e-15

    if spectrum[rank - 1] < eps:
        # When the tested rank is associated with a small eigenvalue, there's
        # no point in computing the log-likelihood: it's going to be very
        # small and won't be the max anyway. Also, it can lead to numerical
        # issues below when computing pa, in particular in log((spectrum[i] -
        # spectrum[j]) because this will take the log of something very small.
        return -xp.inf

    pu = -rank * log(2.0)
    for i in range(1, rank + 1):
        pu += (
            gammaln((n_features - i + 1) / 2.0)
            - log(xp.pi) * (n_features - i + 1) / 2.0
        )

    pl = xp.sum(xp.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.0

    v = max(eps, xp.sum(spectrum[rank:]) / (n_features - rank))
    pv = -log(v) * n_samples * (n_features - rank) / 2.0

    m = n_features * rank - rank * (rank + 1.0) / 2.0
    pp = log(2.0 * xp.pi) * (m + rank) / 2.0

    pa = 0.0
    spectrum_ = xp.asarray(spectrum, copy=True)
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, spectrum.shape[0]):
            pa += log(
                (spectrum[i] - spectrum[j]) * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])
            ) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2.0 - rank * log(n_samples) / 2.0

    return ll

