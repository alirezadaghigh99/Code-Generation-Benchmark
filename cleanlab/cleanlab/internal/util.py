def confusion_matrix(true, pred) -> np.ndarray:
    """Implements a confusion matrix for true labels
    and predicted labels. true and pred MUST BE the same length
    and have the same distinct set of class labels represented.

    Results are identical (and similar computation time) to:
        "sklearn.metrics.confusion_matrix"

    However, this function avoids the dependency on sklearn.

    Parameters
    ----------
    true : np.ndarray 1d
      Contains labels.
      Assumes true and pred contains the same set of distinct labels.

    pred : np.ndarray 1d
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    Returns
    -------
    confusion_matrix : np.ndarray (2D)
      matrix of confusion counts with true on rows and pred on columns."""

    assert len(true) == len(pred)
    true_classes = np.unique(true)
    pred_classes = np.unique(pred)
    K_true = len(true_classes)  # Number of classes in true
    K_pred = len(pred_classes)  # Number of classes in pred
    map_true = dict(zip(true_classes, range(K_true)))
    map_pred = dict(zip(pred_classes, range(K_pred)))

    result = np.zeros((K_true, K_pred))
    for i in range(len(true)):
        result[map_true[true[i]]][map_pred[pred[i]]] += 1

    return result

def value_counts(x, *, num_classes: Optional[int] = None, multi_label=False) -> np.ndarray:
    """Returns an np.ndarray of shape (K, 1), with the
    value counts for every unique item in the labels list/array,
    where K is the number of unique entries in labels.

    Works for both single-labeled and multi-labeled data.

    Parameters
    ----------
    x : list or np.ndarray (one dimensional)
        A list of discrete objects, like lists or strings, for
        example, class labels 'y' when training a classifier.
        e.g. ["dog","dog","cat"] or [1,2,0,1,1,0,2]

    num_classes : int (default: None)
        Setting this fills the value counts for missing classes with zeros.
        For example, if x = [0, 0, 1, 1, 3] then setting ``num_classes=5`` returns
        [2, 2, 0, 1, 0] whereas setting ``num_classes=None`` would return [2, 2, 1]. This assumes
        your labels come from the set [0, 1,... num_classes=1] even if some classes are missing.

    multi_label : bool, optional
      If ``True``, labels should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.
      Assumes all classes in pred_probs.shape[1] are represented in labels.
      The multi-label setting supports classification tasks where an example has 1 or more labels.
      Example of a multi-labeled `labels` input: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], ...]``.
      The major difference in how this is calibrated versus single-label is that
      the total number of errors considered is based on the number of labels,
      not the number of examples. So, the calibrated `confident_joint` will sum
      to the number of total labels."""

    # Efficient method if x is pd.Series, np.ndarray, or list
    if multi_label:
        x = [z for lst in x for z in lst]  # Flatten
    unique_classes, counts = np.unique(x, return_counts=True)

    # Early exit if num_classes is not provided or redundant
    if num_classes is None or num_classes == len(unique_classes):
        return counts

    # Else, there are missing classes
    labels_are_integers = np.issubdtype(np.array(x).dtype, np.integer)
    if labels_are_integers and num_classes <= np.max(unique_classes):
        raise ValueError(f"Required: num_classes > max(x), but {num_classes} <= {np.max(x)}.")

    # Add zero counts for all missing classes in [0, 1,..., num_classes-1]
    total_counts = np.zeros(num_classes, dtype=int)
    # Fill in counts for classes that are present.
    # If labels are integers, unique_classes can be used directly as indices to place counts
    # into the correct positions in total_counts array.
    # If labels are strings, use a slice to fill counts sequentially since strings do not map to indices.
    count_ids = unique_classes if labels_are_integers else slice(len(unique_classes))
    total_counts[count_ids] = counts

    # Return counts with zeros for all missing classes.
    return total_counts

