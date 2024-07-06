def get_normalized_entropy(
    pred_probs: np.ndarray, min_allowed_prob: Optional[float] = None
) -> np.ndarray:
    """Return the normalized entropy of pred_probs.

    Normalized entropy is between 0 and 1. Higher values of entropy indicate higher uncertainty in the model's prediction of the correct label.

    Read more about normalized entropy `on Wikipedia <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Normalized entropy is used in active learning for uncertainty sampling: https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b

    Unlike label-quality scores, entropy only depends on the model's predictions, not the given label.

    Parameters
    ----------
    pred_probs : np.ndarray (shape (N, K))
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class: P(label=k|x)

    min_allowed_prob : float, default: None, deprecated
      Minimum allowed probability value. If not `None` (default),
      entries of `pred_probs` below this value will be clipped to this value.

      .. deprecated:: 2.5.0
         This keyword is deprecated and should be left to the default.
         The entropy is well-behaved even if `pred_probs` contains zeros,
         clipping is unnecessary and (slightly) changes the results.

    Returns
    -------
    entropy : np.ndarray (shape (N, ))
      Each element is the normalized entropy of the corresponding row of ``pred_probs``.

    Raises
    ------
    ValueError
        An error is raised if any of the probabilities is not in the interval [0, 1].
    """
    if np.any(pred_probs < 0) or np.any(pred_probs > 1):
        raise ValueError("All probabilities are required to be in the interval [0, 1].")
    num_classes = pred_probs.shape[1]

    if min_allowed_prob is not None:
        warnings.warn(
            "Using `min_allowed_prob` is not necessary anymore and will be removed.",
            DeprecationWarning,
        )
        pred_probs = np.clip(pred_probs, a_min=min_allowed_prob, a_max=None)

    # Note that dividing by log(num_classes) changes the base of the log which rescales entropy to 0-1 range
    return -np.sum(xlogy(pred_probs, pred_probs), axis=1) / np.log(num_classes)

