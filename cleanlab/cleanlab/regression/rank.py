def get_label_quality_scores(
    labels: ArrayLike,
    predictions: ArrayLike,
    *,
    method: str = "outre",
) -> np.ndarray:
    """
    Returns label quality score for each example in the regression dataset.

    Each score is a continous value in the range [0,1]

    * 1 - clean label (given label is likely correct).
    * 0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : array_like
        Raw labels from original dataset.
        1D array of shape ``(N, )`` containing the given labels for each example (aka. Y-value, response/target/dependent variable), where N is number of examples in the dataset.

    predictions : np.ndarray
        1D array of shape ``(N,)`` containing the predicted label for each example in the dataset.  These should be out-of-sample predictions from a trained regression model, which you can obtain for every example in your dataset via :ref:`cross-validation <pred_probs_cross_val>`.

    method : {"residual", "outre"}, default="outre"
        String specifying which method to use for scoring the quality of each label and identifying which labels appear most noisy.

    Returns
    -------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per example in the dataset.

        Lower scores indicate examples more likely to contain a label issue.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.regression.rank import get_label_quality_scores
    >>> labels = np.array([1,2,3,4])
    >>> predictions = np.array([2,2,5,4.1])
    >>> label_quality_scores = get_label_quality_scores(labels, predictions)
    >>> label_quality_scores
    array([0.00323821, 0.33692597, 0.00191686, 0.33692597])
    """

    # Check if inputs are valid
    labels, predictions = assert_valid_prediction_inputs(
        labels=labels, predictions=predictions, method=method
    )

    scoring_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "residual": _get_residual_score_for_each_label,
        "outre": _get_outre_score_for_each_label,
    }

    scoring_func = scoring_funcs.get(method, None)
    if not scoring_func:
        raise ValueError(
            f"""
            {method} is not a valid scoring method.
            Please choose a valid scoring technique: {scoring_funcs.keys()}.
            """
        )

    # Calculate scores
    label_quality_scores = scoring_func(labels, predictions)
    return label_quality_scores

