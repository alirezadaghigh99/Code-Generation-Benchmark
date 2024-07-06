def _compute_label_quality_scores(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    method: Optional[str] = "objectlab",
    aggregation_weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
    overlapping_label_check: Optional[bool] = True,
    verbose: bool = True,
) -> np.ndarray:
    """Internal function to prune extra bounding boxes and compute label quality scores based on passed in method."""

    pred_probs_prepruned = False
    min_pred_prob = _get_min_pred_prob(predictions)
    aggregation_weights = _get_aggregation_weights(aggregation_weights)

    if threshold is not None:
        predictions = _prune_by_threshold(
            predictions=predictions, threshold=threshold, verbose=verbose
        )
        if np.abs(min_pred_prob - threshold) < 0.001 and threshold > 0:
            pred_probs_prepruned = True  # the provided threshold is the threshold used for pre_pruning the pred_probs during model prediction.
    else:
        threshold = min_pred_prob  # assume model was not pre_pruned if no threshold was provided

    if method == "objectlab":
        scores = _get_subtype_label_quality_scores(
            labels=labels,
            predictions=predictions,
            alpha=ALPHA,
            low_probability_threshold=LOW_PROBABILITY_THRESHOLD,
            high_probability_threshold=HIGH_PROBABILITY_THRESHOLD,
            temperature=TEMPERATURE,
            aggregation_weights=aggregation_weights,
            overlapping_label_check=overlapping_label_check,
        )
    else:
        raise ValueError(
            "Invalid method: '{}' is not a valid method for computing label quality scores. Please use the 'objectlab' method.".format(
                method
            )
        )
    return scores

def get_label_quality_scores(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    aggregation_weights: Optional[Dict[str, float]] = None,
    overlapping_label_check: Optional[bool] = True,
    verbose: bool = True,
) -> np.ndarray:
    """Computes a label quality score for each image of the ``N`` images in the dataset.

    For object detection datasets, the label quality score for an image estimates how likely it has been correctly labeled.
    Lower scores indicate images whose annotation is more likely imperfect.
    Annotators may have mislabeled an image because they:

    - overlooked an object (missing annotated bounding box),
    - chose the wrong class label for an annotated box in the correct location,
    - imperfectly annotated the location/edges of a bounding box.

    Any of these annotation errors should lead to an image with a lower label quality score. This quality score is between 0 and 1.

    - 1 - clean label (given label is likely correct).
    - 0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    verbose : bool, default = True
      Set to ``False`` to suppress all print statements.

    aggregation_weights:
       Optional dictionary to specify weights for aggregating quality scores for subtype of label issue into an overall label quality score for the image.
       Its keys are: "overlooked", "swap", "badloc", and values should be nonnegative weights that sum to 1.
       Increase one of these weights to prioritize images with bounding boxes that were either:
       missing in the annotations (overlooked object), annotated with the wrong class label (class for the object should be swapped to another class), or annotated in a suboptimal location (badly located).

       swapped examples, bad location examples, and overlooked examples.
       It is important to ensure that the weights are non-negative values and that their sum equals 1.0.

    overlapping_label_check : bool, default = True
        If True, boxes annotated with more than one class label have their swap score penalized. Set this to False if you are not concerned when two very similar boxes exist with different class labels in the given annotations.

    Returns
    ---------
    label_quality_scores:
        Array of shape ``(N, )`` of scores between 0 and 1, one per image in the object detection dataset.
        Lower scores indicate images that are more likely mislabeled.
    """
    method = "objectlab"
    probability_threshold = 0.0

    assert_valid_inputs(
        labels=labels,
        predictions=predictions,
        method=method,
        threshold=probability_threshold,
    )
    aggregation_weights = _get_aggregation_weights(aggregation_weights)

    return _compute_label_quality_scores(
        labels=labels,
        predictions=predictions,
        method=method,
        threshold=probability_threshold,
        aggregation_weights=aggregation_weights,
        overlapping_label_check=overlapping_label_check,
        verbose=verbose,
    )

def compute_badloc_box_scores(
    *,
    labels: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[np.ndarray]] = None,
    alpha: Optional[float] = None,
    low_probability_threshold: Optional[float] = None,
    auxiliary_inputs: Optional[List[AuxiliaryTypesDict]] = None,
) -> List[np.ndarray]:
    """
    Returns a numeric score for each annotated bounding box in each image, estimating the likelihood that the edges of this box are not badly located.
    This is a helper method mostly for advanced users.

    A badly located box error is when a box has the correct label but incorrect coordinates so it does not correctly encapsulate the entire object it is for.
    Score per high-confidence predicted bounding box is between 0 and 1, with lower values indicating boxes we are more confident were overlooked in the given label.

    Each image has ``L`` annotated bounding boxes and ``M`` predicted bounding boxes.
    A score is calculated for each predicted box in each of the ``N`` images in dataset.

    Note: ``M`` and ``L`` can be a different values for each image, as the number of annotated and predicted boxes varies.

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    alpha:
        Optional weighting between IoU and Euclidean distance when calculating similarity between predicted and annotated boxes. High alpha means weighting IoU more heavily over Euclidean distance. If no alpha is provided, a good default is used.

    low_probability_threshold:
        Optional minimum probability threshold that determines which predicted boxes are considered when computing badly located scores. If not provided, a good default is used.

    auxiliary_inputs:
        Optional list of ``N`` dictionaries containing keys for sub-parts of label and prediction per image. Useful to minimize computation when computing multiple box scores for a single set of images. For the `i`-th image, `auxiliary_inputs[i]` should contain following keys:

       * pred_labels: np.ndarray
            Array of predicted classes for `i`-th image of shape ``(M,)``.
       * pred_label_probs: np.ndarray
            Array of predicted class probabilities for `i`-th image of shape ``(M,)``.
       * pred_bboxes: np.ndarray
            Array of predicted bounding boxes for `i`-th image of shape ``(M, 4)``.
       * lab_labels: np.ndarray
            Array of given label classed for `i`-th image of shape ``(L,)``.
       * lab_bboxes: np.ndarray
            Array of given label bounding boxes for `i`-th image of shape ``(L, 4)``.
       * similarity_matrix: np.ndarray
            Similarity matrix between labels and predictions `i`-th image.
       * min_possible_similarity: float
            Minimum possible similarity value greater than 0 between labels and predictions for the entire dataset.
    Returns
    ---------
    scores_badloc:
        A list of ``N`` numpy arrays where scores_badloc[i] is an array of size ``L`` badly located scores per annotated box for the `i`-th image.
    """
    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(alpha, low_probability_threshold, None, None)
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(alpha, labels, predictions)

    scores_badloc = []
    for auxiliary_input_dict in auxiliary_inputs:
        scores_badloc_per_box = _compute_badloc_box_scores_for_image(
            alpha=alpha, low_probability_threshold=low_probability_threshold, **auxiliary_input_dict
        )
        scores_badloc.append(scores_badloc_per_box)
    return scores_badloc

def compute_swap_box_scores(
    *,
    labels: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[np.ndarray]] = None,
    alpha: Optional[float] = None,
    high_probability_threshold: Optional[float] = None,
    overlapping_label_check: Optional[bool] = True,
    auxiliary_inputs: Optional[List[AuxiliaryTypesDict]] = None,
) -> List[np.ndarray]:
    """
    Returns a numeric score for each annotated bounding box in each image, estimating the likelihood that the class label for this box was not accidentally swapped with another class.
    This is a helper method mostly for advanced users.

    A swapped box error occurs when a bounding box should be labeled as a class different to what the current label is.
    Score per high-confidence predicted bounding box is between 0 and 1, with lower values indicating boxes we are more confident were overlooked in the given label.

    Each image has ``L`` annotated bounding boxes and ``M`` predicted bounding boxes.
    A score is calculated for each predicted box in each of the ``N`` images in dataset.

    Note: ``M`` and ``L`` can be a different values for each image, as the number of annotated and predicted boxes varies.

    Parameters
    ----------
    labels:
        A list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    predictions:
        A list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model predictions for the `i`-th image.
        Refer to documentation for this argument in :py:func:`find_label_issues <cleanlab.object_detection.filter.find_label_issues>` for further details.

    alpha:
        Optional weighting between IoU and Euclidean distance when calculating similarity between predicted and annotated boxes. High alpha means weighting IoU more heavily over Euclidean distance. If no alpha is provided, a good default is used.

    high_probability_threshold:
        Optional probability threshold that determines which predicted boxes are considered high-confidence when computing overlooked scores. If not provided, a good default is used.

    overlapping_label_check : bool, default = True
        If True, boxes annotated with more than one class label have their swap score penalized. Set this to False if you are not concerned when two very similar boxes exist with different class labels in the given annotations.

    auxiliary_inputs:
        Optional list of ``N`` dictionaries containing keys for sub-parts of label and prediction per image. Useful to minimize computation when computing multiple box scores for a single set of images. For the `i`-th image, `auxiliary_inputs[i]` should contain following keys:

       * pred_labels: np.ndarray
            Array of predicted classes for `i`-th image of shape ``(M,)``.
       * pred_label_probs: np.ndarray
            Array of predicted class probabilities for `i`-th image of shape ``(M,)``.
       * pred_bboxes: np.ndarray
            Array of predicted bounding boxes for `i`-th image of shape ``(M, 4)``.
       * lab_labels: np.ndarray
            Array of given label classed for `i`-th image of shape ``(L,)``.
       * lab_bboxes: np.ndarray
            Array of given label bounding boxes for `i`-th image of shape ``(L, 4)``.
       * similarity_matrix: np.ndarray
            Similarity matrix between labels and predictions `i`-th image.
       * min_possible_similarity: float
            Minimum possible similarity value greater than 0 between labels and predictions for the entire dataset.
    Returns
    ---------
    scores_swap:
        A list of ``N`` numpy arrays where scores_swap[i] is an array of size ``L`` swap scores per annotated box for the `i`-th image.
    """
    (
        alpha,
        low_probability_threshold,
        high_probability_threshold,
        temperature,
    ) = _get_valid_subtype_score_params(alpha, None, high_probability_threshold, None)

    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(alpha, labels, predictions)

    scores_swap = []
    for auxiliary_inputs in auxiliary_inputs:
        scores_swap_per_box = _compute_swap_box_scores_for_image(
            alpha=alpha,
            high_probability_threshold=high_probability_threshold,
            overlapping_label_check=overlapping_label_check,
            **auxiliary_inputs,
        )
        scores_swap.append(scores_swap_per_box)
    return scores_swap

def _get_min_pred_prob(
    predictions: List[np.ndarray],
) -> float:
    """Returns min pred_prob out of all predictions."""
    pred_probs = [1.0]  # avoid calling np.min on empty array.
    for prediction in predictions:
        for class_prediction in prediction:
            pred_probs.extend(list(class_prediction[:, -1]))

    min_pred_prob = np.min(pred_probs)
    return min_pred_prob

