def MetropolisLocal(hilbert, **kwargs) -> MetropolisSampler:
    r"""
    Sampler acting on one local degree of freedom.

    This sampler acts locally only on one local degree of freedom :math:`s_i`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s_N`,
    where :math:`s^\prime_i \neq s_i`.

    The transition probability associated to this
    sampler can be decomposed into two steps:

    1. One of the site indices :math:`i = 1\dots N` is chosen with uniform probability.

    2. Among all the possible (:math:`m - 1`) values that :math:`s^\prime_i` can take,
    one of them is chosen with uniform probability.

    For example, in the case of spin :math:`1/2` particles, :math:`m=2`
    and the possible local values are :math:`s_i = -1,+1`.
    In this case then :class:`MetropolisLocal` is equivalent to flipping a random spin.

    In the case of bosons, with occupation numbers
    :math:`s_i = 0, 1, \dots n_{\mathrm{max}}`, :class:`MetropolisLocal`
    would pick a random local occupation number uniformly between :math:`0`
    and :math:`n_{\mathrm{max}}` except the current :math:`s_i`.

    Args:
        hilbert: The Hilbert space to sample.
        n_chains: The total number of independent Markov chains across all MPI ranks. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    from .rules import LocalRule

    return MetropolisSampler(hilbert, LocalRule(), **kwargs)

