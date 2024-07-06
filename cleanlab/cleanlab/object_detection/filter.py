def find_label_issues(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    *,
    return_indices_ranked_by_score: Optional[bool] = False,
    overlapping_label_check: Optional[bool] = True,
) -> np.ndarray:
    """
    Identifies potentially mislabeled images in an object detection dataset.
    An image is flagged with a label issue if *any* of its bounding boxes appear incorrectly annotated.
    This includes images for which a bounding box: should have been annotated but is missing,
    has been annotated with the wrong class, or has been annotated in a suboptimal location.

    Suppose the dataset has ``N`` images, ``K`` possible class labels.
    If ``return_indices_ranked_by_score`` is ``False``, a boolean mask of length ``N`` is returned,
    indicating whether each image has a label issue (``True``) or not (``False``).
    If ``return_indices_ranked_by_score`` is ``True``, the indices of images flagged with label issues are returned,
    sorted with the most likely-mislabeled images ordered first.

    Parameters
    ----------
    labels:
        Annotated boxes and class labels in the original dataset, which may contain some errors.
        This is a list of ``N`` dictionaries such that ``labels[i]`` contains the given labels for the `i`-th image in the following format:
        ``{'bboxes': np.ndarray((L,4)), 'labels': np.ndarray((L,)), 'image_name': str}`` where ``L`` is the number of annotated bounding boxes
        for the `i`-th image and ``bboxes[l]`` is a bounding box of coordinates in ``[x1,y1,x2,y2]`` format and with given class label ``labels[j]``.
        ``image_name`` is an optional part of the labels that can be used to later refer to specific images.

        Note: Here, ``(x1,y1)`` corresponds to the top-left and ``(x2,y2)`` corresponds to the bottom-right corner of the bounding box with respect to the image matrix [e.g. `XYXY in Keras <https://keras.io/api/keras_cv/bounding_box/formats/>`, `Detectron 2 <https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.visualizer.Visualizer.draw_box>`].

        For more information on proper labels formatting, check out the `MMDetection library <https://mmdetection.readthedocs.io/en/dev-3.x/advanced_guides/customize_dataset.html>`_.

    predictions:
        Predictions output by a trained object detection model.
        For the most accurate results, predictions should be out-of-sample to avoid overfitting, eg. obtained via :ref:`cross-validation <pred_probs_cross_val>`.
        This is a list of ``N`` ``np.ndarray`` such that ``predictions[i]`` corresponds to the model prediction for the `i`-th image.
        For each possible class ``k`` in 0, 1, ..., K-1: ``predictions[i][k]`` is a ``np.ndarray`` of shape ``(M,5)``,
        where ``M`` is the number of predicted bounding boxes for class ``k``. Here the five columns correspond to ``[x1,y1,x2,y2,pred_prob]``,
        where ``[x1,y1,x2,y2]`` are coordinates of the bounding box predicted by the model and ``pred_prob`` is the model's confidence in the predicted class label for this bounding box.

        Note: Here, ``(x1,y1)`` corresponds to the top-left and ``(x2,y2)`` corresponds to the bottom-right corner of the bounding box with respect to the image matrix [e.g. `XYXY in Keras <https://keras.io/api/keras_cv/bounding_box/formats/>`, `Detectron 2 <https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.visualizer.Visualizer.draw_box>`]. The last column, pred_prob, represents the predicted probability that the bounding box contains an object of the class k.

        For more information see the `MMDetection package <https://github.com/open-mmlab/mmdetection>`_ for an example object detection library that outputs predictions in the correct format.

    return_indices_ranked_by_score:
        Determines what is returned by this method (see description of return value for details).

    overlapping_label_check : bool, default = True
       If True, boxes annotated with more than one class label have their swap score penalized.  Set this to False if you are not concerned when two very similar boxes exist with different class labels in the given annotations.


    Returns
    -------
    label_issues : np.ndarray
        Specifies which images are identified to have a label issue.
        If ``return_indices_ranked_by_score = False``, this function returns a boolean mask of length ``N`` (``True`` entries indicate which images have label issue).
        If ``return_indices_ranked_by_score = True``, this function returns a (shorter) array of indices of images with label issues, sorted by how likely the image is mislabeled.

        More precisely, indices are sorted by image label quality score calculated via :py:func:`object_detection.rank.get_label_quality_scores <cleanlab.object_detection.rank.get_label_quality_scores>`.
    """
    scoring_method = "objectlab"

    assert_valid_inputs(
        labels=labels,
        predictions=predictions,
        method=scoring_method,
    )

    is_issue = _find_label_issues(
        labels,
        predictions,
        scoring_method=scoring_method,
        return_indices_ranked_by_score=return_indices_ranked_by_score,
        overlapping_label_check=overlapping_label_check,
    )

    return is_issue

def _calculate_true_positives_false_positives(
    pred_bboxes: np.ndarray,
    lab_bboxes: np.ndarray,
    iou_threshold: Optional[float] = 0.5,
    return_false_negative: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates true positives (TP) and false positives (FP) for object detection tasks.
    It takes predicted bounding boxes, ground truth bounding boxes, and an optional Intersection over Union (IoU) threshold as inputs.
    If return_false_negative is True, it returns an array of False negatives as well.
    """
    num_preds = pred_bboxes.shape[0]
    num_labels = lab_bboxes.shape[0]
    num_scales = 1
    true_positives = np.zeros((num_scales, num_preds), dtype=np.float32)
    false_positives = np.zeros((num_scales, num_preds), dtype=np.float32)

    if lab_bboxes.shape[0] == 0:
        false_positives[...] = 1
        if return_false_negative:
            return true_positives, false_positives, np.array([], dtype=np.float32)
        else:
            return true_positives, false_positives
    ious = _get_overlap_matrix(pred_bboxes, lab_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sorted_indices = np.argsort(-pred_bboxes[:, -1])
    is_covered = np.zeros(num_labels, dtype=bool)
    for index in sorted_indices:
        if ious_max[index] >= iou_threshold:
            matching_label = ious_argmax[index]
            if not is_covered[matching_label]:
                is_covered[matching_label] = True
                true_positives[0, index] = 1
            else:
                false_positives[0, index] = 1
        else:
            false_positives[0, index] = 1
    if return_false_negative:
        false_negatives = np.zeros((num_scales, num_labels), dtype=np.float32)
        for label_index in range(num_labels):
            if not is_covered[label_index]:
                false_negatives[0, label_index] = 1
        return true_positives, false_positives, false_negatives
    return true_positives, false_positives

