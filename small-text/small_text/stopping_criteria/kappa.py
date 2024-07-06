def _adapted_cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None):
    """Extends cohen's kappa by intercepting the special case of a perfect agreement, which results in a
    division by zero when adhering to the original formula. In case of a perfect agreement `1.0` is returned, otherwise
    the call is delegated to the `cohen_kappa_score()` implementation in scikit-learn.

    .. seealso::
       Dcumentation of the underlying `cohen_kappa_score()` method in scikit-learn.
           https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
    """
    if np.array_equal(y1, y2):
        return 1.0
    else:
        return cohen_kappa_score(
            y1,
            y2,
            labels=labels,
            weights=weights,
            sample_weight=sample_weight
        )

