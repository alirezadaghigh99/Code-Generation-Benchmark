def statistics(data, batch_size=32):
    r"""
    Returns statistics of a given array (or matrix, see below) containing a stream of data.
    This is particularly useful to analyze Markov Chain data, but it can be used
    also for other type of time series.
    Assumes same shape on all MPI processes.

    Args:
        data (vector or matrix): The input data. It can be real or complex valued.
            * if a vector, it is assumed that this is a time series of data (not necessarily independent);
            * if a matrix, it is assumed that that rows :code:`data[i]` contain independent time series.

    Returns:
       Stats: A dictionary-compatible class containing the
             average (:code:`.mean`, :code:`["Mean"]`),
             variance (:code:`.variance`, :code:`["Variance"]`),
             the Monte Carlo standard error of the mean (:code:`error_of_mean`, :code:`["Sigma"]`),
             an estimate of the autocorrelation time (:code:`tau_corr`, :code:`["TauCorr"]`), and the
             Gelman-Rubin split-Rhat diagnostic (:code:`.R_hat`, :code:`["R_hat"]`).

             These properties can be accessed both the attribute and the dictionary-style syntax
             (both indicated above).

             The split-Rhat diagnostic is based on comparing intra-chain and inter-chain
             statistics of the sample and is thus only available for 2d-array inputs where
             the rows are independently sampled MCMC chains. In an ideal MCMC samples,
             R_hat should be 1.0. If it deviates from this value too much, this indicates
             MCMC convergence issues. Thresholds such as R_hat > 1.1 or even R_hat > 1.01 have
             been suggested in the literature for when to discard a sample. (See, e.g.,
             Gelman et al., `Bayesian Data Analysis <http://www.stat.columbia.edu/~gelman/book/>`_,
             or Vehtari et al., `arXiv:1903.08008 <https://arxiv.org/abs/1903.08008>`_.)
    """
    return _statistics(data, batch_size)

