def find_overlapping_classes(
    labels=None,
    pred_probs=None,
    *,
    asymmetric=False,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
) -> pd.DataFrame:
    """Returns the pairs of classes that are often mislabeled as one another.
    Consider merging the top pairs of classes returned by this method each into a single class.
    If the dataset is labeled by human annotators, consider clearly defining the
    difference between the classes prior to having annotators label the data.

    This method provides two scores in the Pandas DataFrame that is returned:

    * **Num Overlapping Examples**: The number of examples where the two classes overlap
    * **Joint Probability**: `(num overlapping examples / total number of examples in the dataset`).

    This method works by providing any one (and only one) of the following inputs:

    1. ``labels`` and ``pred_probs``, or
    2. ``joint`` and ``num_examples``, or
    3. ``confident_joint``

    Only provide **exactly one of the above input options**, do not provide a combination.

    This method uses the joint distribution of noisy and true labels to compute ontological
    issues via the approach published in `Northcutt et al.,
    2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Examples
    --------
    >>> from cleanlab.dataset import find_overlapping_classes
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_predict
    >>> data, labels = get_data_labels_from_dataset()
    >>> yourFavoriteModel = LogisticRegression()
    >>> pred_probs = cross_val_predict(yourFavoriteModel, data, labels, cv=3, method="predict_proba")
    >>> df = find_overlapping_classes(labels=labels, pred_probs=pred_probs)

    Note
    ----
    The joint distribution of noisy and true labels is asymmetric, and therefore the joint
    probability ``p(given="vehicle", true="truck") != p(true="truck", given="vehicle")``.
    This is intuitive. Images of trucks (true label) are much more likely to be labeled as a car
    (given label) than images of cars (true label) being frequently mislabeled as truck (given
    label). cleanlab takes these differences into account for you automatically via the joint
    distribution. If you do not want this behavior, simply set ``asymmetric=False``.

    This method estimates how often the annotators confuse two classes.
    This differs from just using a similarity matrix or confusion matrix,
    as these summarize characteristics of the predictive model rather than the data labelers (i.e. annotators).
    Instead, this method works even if the model that generated `pred_probs` tends to be more confident in some classes than others.

    Parameters
    ----------
    labels : np.ndarray or list, optional
      An array_like (of length N) of noisy labels for the classification dataset, i.e. some labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.
      All the classes (0, 1, ..., and K-1) should be present in ``labels``, such that
      ``len(set(labels)) == pred_probs.shape[1]`` for standard multi-class classification with single-labeled data (e.g. ``labels =  [1,0,2,1,1,0...]``).
      For multi-label classification where each example can belong to multiple classes (e.g. ``labels = [[1,2],[1],[0],[],...]``),
      your labels should instead satisfy: ``len(set(k for l in labels for k in l)) == pred_probs.shape[1])``.

    pred_probs : np.ndarray, optional
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1. `pred_probs` should have been computed using 3 (or
      higher) fold cross-validation.

    asymmetric : bool, optional
      If ``asymmetric=True``, returns separate estimates for both pairs (class1, class2) and (class2, class1). Use this
      for finding "is a" relationships where for example "class1 is a class2".
      In this case, num overlapping examples counts the number of examples that have been labeled as class1 which should actually have been labeled as class2.
      If ``asymmetric=False``, the pair (class1, class2) will only be returned once with an arbitrary order.
      In this case, their estimated score is the sum: ``score(class1, class2) + score(class2, class1))``.

    class_names : Iterable[str]
        A list or other iterable of the string class names. The list should be in the order that
        matches the class indices. So if class 0 is 'dog' and class 1 is 'cat', then
        ``class_names = ['dog', 'cat']``.

    num_examples : int or None, optional
        The number of examples in the dataset, i.e. ``len(labels)``. You only need to provide this if
        you use this function with the joint, e.g. ``find_overlapping_classes(joint=joint)``, otherwise
        this is automatically computed via ``sum(confident_joint)`` or ``len(labels)``.

    joint : np.ndarray, optional
        An array of shape ``(K, K)``, where K is the number of classes,
        representing the estimated joint distribution of the noisy labels and
        true labels. The sum of all entries in this matrix must be 1 (valid
        probability distribution). Each entry in the matrix captures the co-occurence joint
        probability of a true label and a noisy label, i.e. ``p(noisy_label=i, true_label=j)``.
        **Important**. If you input the joint, you must also input `num_examples`.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    Returns
    -------
    overlapping_classes : pd.DataFrame
        Pandas DataFrame with columns "Class Index A", "Class Index B",
        "Num Overlapping Examples", "Joint Probability" and a description of each below.
        Each row corresponds to a pair of classes.

        * *Class Index A*: the index of a class in 0, 1, ..., K-1.
        * *Class Index B*: the index of a different class (from Class A) in 0, 1, ..., K-1.
        * *Num Overlapping Examples*: estimated number of labels overlapping between the two classes.
        * *Joint Probability*: the *Num Overlapping Examples* divided by the number of examples in the dataset.

        By default, the DataFrame is ordered by "Joint Probability" descending.
    """

    def _2d_matrix_to_row_column_value_list(matrix):
        """Create a list<tuple> [(row_index, col_index, value)] representation of matrix.

        Parameters
        ----------
        matrix : np.ndarray<float>
            Any valid np.ndarray 2-d dimensional matrix.

        Returns
        -------
        list<tuple>
            A [(row_index, col_index, value)] representation of matrix.
        """

        return [(*i, v) for i, v in np.ndenumerate(matrix)]

    if multi_label:
        raise ValueError(
            "For multilabel data, please instead call: multilabel_classification.dataset.common_multilabel_issues()"
        )

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels, confident_joint=confident_joint)
    if asymmetric:
        rcv_list = _2d_matrix_to_row_column_value_list(joint)
        # Remove diagonal elements
        rcv_list = [tup for tup in rcv_list if tup[0] != tup[1]]
    else:  # symmetric
        # Sum the upper and lower triangles and remove the lower triangle and the diagonal
        sym_joint = np.triu(joint) + np.tril(joint).T
        rcv_list = _2d_matrix_to_row_column_value_list(sym_joint)
        # Provide values only in (the upper triangle) of the matrix.
        rcv_list = [tup for tup in rcv_list if tup[0] < tup[1]]
    df = pd.DataFrame(rcv_list, columns=["Class Index A", "Class Index B", "Joint Probability"])
    num_overlapping = (df["Joint Probability"] * num_examples).round().astype(int)
    df.insert(loc=2, column="Num Overlapping Examples", value=num_overlapping)
    if class_names is not None:
        df.insert(
            loc=0, column="Class Name A", value=df["Class Index A"].apply(lambda x: class_names[x])
        )
        df.insert(
            loc=1, column="Class Name B", value=df["Class Index B"].apply(lambda x: class_names[x])
        )
    return df.sort_values(by="Joint Probability", ascending=False).reset_index(drop=True)

