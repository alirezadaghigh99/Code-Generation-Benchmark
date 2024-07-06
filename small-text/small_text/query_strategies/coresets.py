def greedy_coreset(x, indices_unlabeled, indices_labeled, n, distance_metric='cosine',
                   batch_size=100, normalized=False):
    """Computes a greedy coreset [SS17]_ over `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        A matrix of row-wise vector representations.
    indices_unlabeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    indices_labeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    n : int
        Size of the coreset (in number of instances).
    distance_metric : {'cosine', 'euclidean'}
        Distance metric to be used.
    batch_size : int
        Batch size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [SS17] Ozan Sener and Silvio Savarese. 2017.
       Active Learning for Convolutional Neural Networks: A Core-Set Approach.
       In International Conference on Learning Representations 2018 (ICLR 2018).
    """
    _check_coreset_size(x, n)

    num_batches = int(np.ceil(indices_unlabeled.shape[0] / batch_size))
    ind_new = []

    if distance_metric == 'cosine':
        dist_func = _cosine_distance
    elif distance_metric == 'euclidean':
        dist_func = _euclidean_distance
    else:
        raise ValueError(f'Invalid distance metric: {distance_metric}. '
                         f'Possible values: {_DISTANCE_METRICS}')

    for _ in range(n):
        indices_s = np.concatenate([indices_labeled, ind_new]).astype(np.int64)
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x[indices_unlabeled], num_batches, axis=0):

            dist = dist_func(batch, x[indices_s], normalized=normalized)

            sims_batch = np.amin(dist, axis=1)
            dists = np.append(dists, sims_batch)

        dists[ind_new] = -np.inf
        index_new = np.argmax(dists)

        ind_new.append(index_new)

    return np.array(ind_new)

