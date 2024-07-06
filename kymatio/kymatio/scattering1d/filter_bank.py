def compute_xi_max(Q):
    """
    Computes the maximal xi to use for the Morlet family, depending on Q.

    Parameters
    ----------
    Q : int
        number of wavelets per octave (integer >= 1)

    Returns
    -------
    xi_max : float
        largest frequency of the wavelet frame.
    """
    xi_max = max(1. / (1. + math.pow(2., 3. / Q)), 0.35)
    return xi_max

