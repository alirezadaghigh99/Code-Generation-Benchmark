def _get_ood_predictions_scores(
    pred_probs: np.ndarray,
    *,
    labels: Optional[LabelLike] = None,
    confident_thresholds: Optional[np.ndarray] = None,
    adjust_pred_probs: bool = True,
    method: str = "entropy",
    M: int = 100,
    gamma: float = 0.1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return an OOD (out of distribution) score for each example based on it pred_prob values.

    Parameters
    ----------
    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted probabilities,
      `pred_probs` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.

    confident_thresholds : np.ndarray, default = None
      For details, see key `confident_thresholds` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

    labels : array_like, optional
      `labels` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.

    adjust_pred_probs : bool, True
      Account for class imbalance in the label-quality scoring.
      For details, see key `adjust_pred_probs` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

    method : {"entropy", "least_confidence", "gen"}, default="entropy"
      Which method to use for computing outlier scores based on pred_probs.
      For details see key `method` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.

    M : int, default=100
      For GEN method only. Hyperparameter that controls the number of top classes to consider when calculating OOD scores.

    gamma : float, default=0.1
      For GEN method only. Hyperparameter that controls the weight of the second term in the GEN score.


    Returns
    -------
    ood_predictions_scores : Tuple[np.ndarray, Optional[np.ndarray]]
      Returns a tuple. First element is array of `ood_predictions_scores` and second is an np.ndarray of `confident_thresholds` or None is 'confident_thresholds' is not calculated.
    """
    valid_methods = (
        "entropy",
        "least_confidence",
        "gen",
    )

    if (confident_thresholds is not None or labels is not None) and not adjust_pred_probs:
        warnings.warn(
            "OOD scores are not adjusted with confident thresholds. If scores need to be adjusted set "
            "params['adjusted_pred_probs'] = True. Otherwise passing in confident_thresholds and/or labels does not change "
            "score calculation.",
            UserWarning,
        )

    if adjust_pred_probs:
        if confident_thresholds is None:
            if labels is None:
                raise ValueError(
                    "Cannot calculate adjust_pred_probs without labels. Either pass in labels parameter or set "
                    "params['adjusted_pred_probs'] = False. "
                )
            labels = labels_to_array(labels)
            assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)
            confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=False)

        pred_probs = _subtract_confident_thresholds(
            None, pred_probs, multi_label=False, confident_thresholds=confident_thresholds
        )

    # Scores are flipped so ood scores are closer to 0. Scores reflect confidence example is in-distribution.
    if method == "entropy":
        ood_predictions_scores = 1.0 - get_normalized_entropy(pred_probs)
    elif method == "least_confidence":
        ood_predictions_scores = pred_probs.max(axis=1)
    elif method == "gen":
        if pred_probs.shape[1] < M:  # pragma: no cover
            warnings.warn(
                f"GEN with the default hyperparameter settings is intended for datasets with at least {M} classes. You can adjust params['M'] according to the number of classes in your dataset.",
                UserWarning,
            )
        probs = softmax(pred_probs, axis=1)
        probs_sorted = np.sort(probs, axis=1)[:, -M:]
        ood_predictions_scores = (
            1 - np.sum(probs_sorted**gamma * (1 - probs_sorted) ** (gamma), axis=1) / M
        )  # Use 1 + original gen score/M to make the scores lie in 0-1
    else:
        raise ValueError(
            f"""
            {method} is not a valid OOD scoring method!
            Please choose a valid scoring_method: {valid_methods}
            """
        )

    return (
        ood_predictions_scores,
        confident_thresholds,
    )

