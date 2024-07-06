def merge_probs(
    probs: npt.NDArray["np.floating[T]"], maps: List[int]
) -> npt.NDArray["np.floating[T]"]:
    """
    Merges model-predictive probabilities with desired mapping

    Parameters
    ----------
    probs:
        A 2D np.array of shape `(N, K)`, where N is the number of tokens, and K is the number of classes for the model

    maps:
        a list of mapped index, such that the probability of the token being in the i'th class is mapped to the
        `maps[i]` index. If `maps[i] == -1`, the i'th column of `probs` is ignored. If `np.any(maps == -1)`, the
        returned probability is re-normalized.

    Returns
    ---------
    probs_merged:
        A 2D np.array of shape ``(N, K')``, where `K'` is the number of new classes. Probabilities are merged and
        re-normalized if necessary.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.internal.token_classification_utils import merge_probs
    >>> probs = np.array([
    ...     [0.55, 0.0125, 0.0375, 0.1, 0.3],
    ...     [0.1, 0.8, 0, 0.075, 0.025],
    ... ])
    >>> maps = [0, 1, 1, 2, 2]
    >>> merge_probs(probs, maps)
    array([[0.55, 0.05, 0.4 ],
           [0.1 , 0.8 , 0.1 ]])
    """
    old_classes = probs.shape[1]
    map_size = np.max(maps) + 1
    probs_merged = np.zeros([len(probs), map_size], dtype=probs.dtype.type)

    for i in range(old_classes):
        if maps[i] >= 0:
            probs_merged[:, maps[i]] += probs[:, i]
    if -1 in maps:
        row_sums = probs_merged.sum(axis=1)
        probs_merged /= row_sums[:, np.newaxis]
    return probs_merged

