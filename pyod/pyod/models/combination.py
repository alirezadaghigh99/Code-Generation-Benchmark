def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False,
        random_state=None):
    """Average of Maximum - An ensemble method for combining multiple
    estimators. See :cite:`aggarwal2015theoretical` for details.

    First dividing estimators into subgroups, take the maximum score as the
    subgroup score. Finally, take the average of all subgroup outlier scores.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        The score matrix outputted from various estimators

    n_buckets : int, optional (default=5)
        The number of subgroups to build

    method : str, optional (default='static')
        {'static', 'dynamic'}, if 'dynamic', build subgroups
        randomly with dynamic bucket size.

    bootstrap_estimators : bool, optional (default=False)
        Whether estimators are drawn with replacement.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator
        is the RandomState instance used by `np.random`.

    Returns
    -------
    combined_scores : Numpy array of shape (n_samples,)
        The combined outlier scores.

    """

    return combo_aom(scores, n_buckets, method, bootstrap_estimators,
                     random_state)def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False,
        random_state=None):
    """Maximization of Average - An ensemble method for combining multiple
    estimators. See :cite:`aggarwal2015theoretical` for details.

    First dividing estimators into subgroups, take the average score as the
    subgroup score. Finally, take the maximization of all subgroup outlier
    scores.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        The score matrix outputted from various estimators

    n_buckets : int, optional (default=5)
        The number of subgroups to build

    method : str, optional (default='static')
        {'static', 'dynamic'}, if 'dynamic', build subgroups
        randomly with dynamic bucket size.

    bootstrap_estimators : bool, optional (default=False)
        Whether estimators are drawn with replacement.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator
        is the RandomState instance used by `np.random`.

    Returns
    -------
    combined_scores : Numpy array of shape (n_samples,)
        The combined outlier scores.

    """
    return combo_moa(scores, n_buckets, method, bootstrap_estimators,
                     random_state)