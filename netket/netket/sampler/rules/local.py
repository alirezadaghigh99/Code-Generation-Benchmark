class LocalRule(MetropolisRule):
    r"""
    A transition rule acting on the local degree of freedom.

    This transition acts locally only on one local degree of freedom :math:`s_i`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s_N`,
    where :math:`s^\prime_i \neq s_i`.

    The transition probability associated to this
    sampler can be decomposed into two steps:

    1. One of the site indices :math:`i = 1\dots N` is chosen
    with uniform probability.
    2. Among all the possible (:math:`m`) values that :math:`s_i` can take,
    one of them is chosen with uniform probability.
    """

    def transition(rule, sampler, machine, parameters, state, key, σ):
        key1, key2 = jax.random.split(key, 2)

        n_chains = σ.shape[0]
        hilb = sampler.hilbert

        indxs = jax.random.randint(key1, shape=(n_chains,), minval=0, maxval=hilb.size)
        σp, _ = flip_state(hilb, key2, σ, indxs)

        return σp, None

    def __repr__(self):
        return "LocalRule()"

