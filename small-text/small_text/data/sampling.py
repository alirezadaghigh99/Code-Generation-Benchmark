def stratified_sampling(y, n_samples=10, enforce_min_occurrence=True):
    """
    Performs a stratified random sampling.

    Parameters
    ----------
    y : numpy.ndarray or scipy.sparse.csr_matrix
        Dense or sparse matrix of of labels.
    n_samples : int, default=10
        Number of indices to sample.
    enforce_min_occurrence : bool, default=True
        Ensures that at least one sample from each class (provided it is present in the data)
        is included in the stratified sample.

    Returns
    -------
    indices : numpy.ndarray
        Indices of the stratified subset.

    Notes
    -----
    Only useful for experimental simulations (Requires label knowledge).
    """
    _assert_sample_size(y, n_samples)

    # TODO: check for gaps in given labels

    # num classes according to the labels
    num_classes = np.max(y) + 1

    counts = _get_class_histogram(y, num_classes)
    expected_samples_per_class = np.floor(counts * (float(n_samples) / counts.sum())).astype(int)

    if enforce_min_occurrence and expected_samples_per_class.min() == 0:
        if n_samples > num_classes and np.unique(y).shape[0] == num_classes:  # feasibility check
            expected_samples_per_class += 1

            num_excessive_samples = expected_samples_per_class.sum() - n_samples
            class_indices = np.arange(counts.shape[0])[expected_samples_per_class > 1]
            round_robin_index = 0
            for i in range(num_excessive_samples):

                while expected_samples_per_class[class_indices[round_robin_index]] <= 1:
                    round_robin_index += 1
                    round_robin_index %= class_indices.shape[0]

                expected_samples_per_class[class_indices[round_robin_index]] -= 1

                class_indices = np.arange(counts.shape[0])[expected_samples_per_class > 1]
                assert expected_samples_per_class[class_indices].sum() > 0

    return _random_sampling(n_samples, num_classes, expected_samples_per_class, counts, y)

def balanced_sampling(y, n_samples=10):
    """
    Performs a class-balanced random sampling.

    If `n_samples` is not divisible by the number of classes, a number of samples equal to the
    remainder will be sampled randomly among the classes.

    Parameters
    ----------
    y : list of int or numpy.ndarray
        List of labels.
    n_samples : int, default=10
        Number of indices to sample.

    Returns
    -------
    indices : numpy.ndarray
        Indices of the stratified subset.

    Notes
    -----
    Only useful for experimental simulations (Requires label knowledge).
    """
    _assert_sample_size(y, n_samples)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # TODO: check for gaps in given labels

    # num classes according to the labels
    num_classes = np.max(y) + 1
    # num classes encountered
    num_classes_present = len(np.unique(y))

    counts = _get_class_histogram(y, num_classes)
    expected_samples_per_class = np.array([int(n_samples / num_classes_present)] * num_classes)

    return _random_sampling(n_samples, num_classes, expected_samples_per_class, counts, y)

