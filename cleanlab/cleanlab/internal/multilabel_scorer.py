def exponential_moving_average(
    s: np.ndarray,
    *,
    alpha: Optional[float] = None,
    axis: int = 1,
    **_,
) -> np.ndarray:
    r"""Exponential moving average (EMA) score aggregation function.

    For a score vector s = (s_1, ..., s_K) with K scores, the values
    are sorted in *descending* order and the exponential moving average
    of the last score is calculated, denoted as EMA_K according to the
    note below.

    Note
    ----

    The recursive formula for the EMA at step :math:`t = 2, ..., K` is:

    .. math::

        \text{EMA}_t = \alpha \cdot s_t + (1 - \alpha) \cdot \text{EMA}_{t-1}, \qquad 0 \leq \alpha \leq 1

    We set :math:`\text{EMA}_1 = s_1` as the largest score in the sorted vector s.

    :math:`\alpha` is the "forgetting factor" that gives more weight to the
    most recent scores, and successively less weight to the previous scores.

    Parameters
    ----------
    s :
        Scores to be transformed.

    alpha :
        Discount factor that determines the weight of the previous EMA score.
        Higher alpha means that the previous EMA score has a lower weight while
        the current score has a higher weight.

        Its value must be in the interval [0, 1].

        If alpha is None, it is set to 2 / (K + 1) where K is the number of scores.

    axis :
        Axis along which the scores are sorted.

    Returns
    -------
    s_ema :
        Exponential moving average score.

    Examples
    --------
    >>> from cleanlab.internal.multilabel_scorer import exponential_moving_average
    >>> import numpy as np
    >>> s = np.array([[0.1, 0.2, 0.3]])
    >>> exponential_moving_average(s, alpha=0.5)
    np.array([0.175])
    """
    K = s.shape[1]
    s_sorted = np.fliplr(np.sort(s, axis=axis))
    if alpha is None:
        # One conventional choice for alpha is 2/(K + 1), where K is the number of periods in the moving average.
        alpha = float(2 / (K + 1))
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in the interval [0, 1], got {alpha}")
    s_T = s_sorted.T
    s_ema, s_next = s_T[0], s_T[1:]
    for s_i in s_next:
        s_ema = alpha * s_i + (1 - alpha) * s_ema
    return s_ema

def _get_split_generator(labels, cv):
    _, multilabel_ids = np.unique(labels, axis=0, return_inverse=True)
    split_generator = cv.split(X=multilabel_ids, y=multilabel_ids)
    return split_generator

