def compute_confident_joint(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    thresholds: Optional[Union[np.ndarray, list]] = None,
    calibrate: bool = True,
    multi_label: bool = False,
    return_indices_of_off_diagonals: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Estimates the confident counts of latent true vs observed noisy labels
    for the examples in our dataset. This array of shape ``(K, K)`` is called the **confident joint**
    and contains counts of examples in every class, confidently labeled as every other class.
    These counts may subsequently be used to estimate the joint distribution of true and noisy labels
    (by normalizing them to frequencies).

    Important: this function assumes that `pred_probs` are out-of-sample
    holdout probabilities. This can be :ref:`done with cross validation <pred_probs_cross_val>`. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    thresholds : array_like, optional
      An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
      probabilities, used to determine the cutoff probability necessary to
      consider an example as a given class label (see `Northcutt et al.,
      2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
      3.1, Equation 2).

      This is for advanced users only. If not specified, these are computed
      for you automatically. If an example has a predicted probability
      greater than this threshold, it is counted as having true_label =
      k. This is not used for pruning/filtering, only for estimating the
      noise rates using confident counts.

    calibrate : bool, default=True
        Calibrates confident joint estimate ``P(label=i, true_label=j)`` such that
        ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)``.
        When ``calibrate=True``, this method returns an estimate of
        the latent true joint counts of noisy and true labels.

    multi_label : bool, optional
      If ``True``, this is multi-label classification dataset (where each example can belong to more than one class)
      rather than a regular (multi-class) classifiction dataset.
      In this case, `labels` should be an iterable (e.g. list) of iterables (e.g. ``List[List[int]]``),
      containing the list of classes to which each example belongs, instead of just a single class.
      Example of `labels` for a multi-label classification dataset: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], [], ...]``.

    return_indices_of_off_diagonals : bool, optional
      If ``True``, returns indices of examples that were counted in off-diagonals
      of confident joint as a baseline proxy for the label issues. This
      sometimes works as well as ``filter.find_label_issues(confident_joint)``.


    Returns
    -------
    confident_joint_counts : np.ndarray
      An array of shape ``(K, K)`` representing counts of examples
      for which we are confident about their given and true label (if `multi_label` is False).
      If `multi_label` is True,
      this array instead has shape ``(K, 2, 2)`` representing a one-vs-rest format for the  confident joint, where for each class `c`:
      Entry ``(c, 0, 0)`` in this one-vs-rest array is the number of examples whose noisy label contains `c` confidently identified as truly belonging to class `c` as well.
      Entry ``(c, 1, 0)`` in this one-vs-rest array is the number of examples whose noisy label contains `c` confidently identified as not actually belonging to class `c`.
      Entry ``(c, 0, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as truly belonging to class `c`.
      Entry ``(c, 1, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as actually not belonging to class `c` as well.


      Note
      ----
      If `return_indices_of_off_diagonals` is set as True, this function instead returns a tuple `(confident_joint, indices_off_diagonal)`
      where `indices_off_diagonal` is a list of arrays and each array contains the indices of examples counted in off-diagonals of confident joint.

    Note
    ----
    We provide a for-loop based simplification of the confident joint
    below. This implementation is not efficient, not used in practice, and
    not complete, but covers the gist of how the confident joint is computed:

    .. code:: python

        # Confident examples are those that we are confident have true_label = k
        # Estimate (K, K) matrix of confident examples with label = k_s and true_label = k_y
        cj_ish = np.zeros((K, K))
        for k_s in range(K): # k_s is the class value k of noisy labels `s`
            for k_y in range(K): # k_y is the (guessed) class k of true_label k_y
                cj_ish[k_s][k_y] = sum((pred_probs[:,k_y] >= (thresholds[k_y] - 1e-8)) & (labels == k_s))

    The following is a vectorized (but non-parallelized) implementation of the
    confident joint, again slow, using for-loops/simplified for understanding.
    This implementation is 100% accurate, it's just not optimized for speed.

    .. code:: python

        confident_joint = np.zeros((K, K), dtype = int)
        for i, row in enumerate(pred_probs):
            s_label = labels[i]
            confident_bins = row >= thresholds - 1e-6
            num_confident_bins = sum(confident_bins)
            if num_confident_bins == 1:
                confident_joint[s_label][np.argmax(confident_bins)] += 1
            elif num_confident_bins > 1:
                confident_joint[s_label][np.argmax(row)] += 1
    """

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")

        return _compute_confident_joint_multi_label(
            labels=labels,
            pred_probs=pred_probs,
            thresholds=thresholds,
            calibrate=calibrate,
            return_indices_of_off_diagonals=return_indices_of_off_diagonals,
        )

    # labels needs to be a numpy array
    labels = np.asarray(labels)

    # Estimate the probability thresholds for confident counting
    if thresholds is None:
        # P(we predict the given noisy label is k | given noisy label is k)
        thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)
    thresholds = np.asarray(thresholds)

    # Compute confident joint (vectorized for speed).

    # pred_probs_bool is a bool matrix where each row represents a training example as a boolean vector of
    # size num_classes, with True if the example confidently belongs to that class and False if not.
    pred_probs_bool = pred_probs >= thresholds - 1e-6
    num_confident_bins = pred_probs_bool.sum(axis=1)
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1
    pred_probs_argmax = pred_probs.argmax(axis=1)
    # Note that confident_argmax is meaningless for rows of all False
    confident_argmax = pred_probs_bool.argmax(axis=1)
    # For each example, choose the confident class (greater than threshold)
    # When there is 2+ confident classes, choose the class with largest prob.
    true_label_guess = np.where(
        more_than_one_confident,
        pred_probs_argmax,
        confident_argmax,
    )
    # true_labels_confident omits meaningless all-False rows
    true_labels_confident = true_label_guess[at_least_one_confident]
    labels_confident = labels[at_least_one_confident]
    confident_joint = confusion_matrix(
        y_true=true_labels_confident,
        y_pred=labels_confident,
        labels=range(pred_probs.shape[1]),
    ).T  # Guarantee at least one correctly labeled example is represented in every class
    np.fill_diagonal(confident_joint, confident_joint.diagonal().clip(min=1))
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)

    if return_indices_of_off_diagonals:
        true_labels_neq_given_labels = true_labels_confident != labels_confident
        indices = np.arange(len(labels))[at_least_one_confident][true_labels_neq_given_labels]

        return confident_joint, indices

    return confident_joint

def estimate_joint(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    confident_joint: Optional[np.ndarray] = None,
    multi_label: bool = False,
) -> np.ndarray:
    """
    Estimates the joint distribution of label noise ``P(label=i, true_label=j)`` guaranteed to:

    * Sum to 1
    * Satisfy ``np.sum(joint_estimate, axis = 1) == p(labels)``

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    confident_joint : np.ndarray, optional
      Array of estimated class label error statisics used for identifying label issues,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      If not provided, it is internally computed from the given (noisy) `labels` and `pred_probs`.

    multi_label : bool, optional
      If ``False``, dataset is for regular (multi-class) classification, where each example belongs to exactly one class.
      If ``True``, dataset is for multi-label classification, where each example can belong to multiple classes.
      See documentation of `~cleanlab.count.compute_confident_joint` for details.

    Returns
    -------
    confident_joint_distribution : np.ndarray
      An array of shape ``(K, K)`` representing an
      estimate of the true joint distribution of noisy and true labels (if `multi_label` is False).
      If `multi_label` is True, an array of shape ``(K, 2, 2)`` representing an
      estimate of the true joint distribution of noisy and true labels for each class in a one-vs-rest fashion.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).
    """

    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            labels,
            pred_probs,
            calibrate=True,
            multi_label=multi_label,
        )
    else:
        if labels is not None:
            calibrated_cj = calibrate_confident_joint(
                confident_joint, labels, multi_label=multi_label
            )
        else:
            calibrated_cj = confident_joint

    assert isinstance(calibrated_cj, np.ndarray)
    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _estimate_joint_multilabel(
                labels=labels, pred_probs=pred_probs, confident_joint=confident_joint
            )
    else:
        return calibrated_cj / np.clip(float(np.sum(calibrated_cj)), a_min=TINY_VALUE, a_max=None)

def estimate_latent(
    confident_joint: np.ndarray,
    labels: np.ndarray,
    *,
    py_method: str = "cnt",
    converge_latent_estimates: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the latent prior ``p(y)``, the noise matrix ``P(labels|y)`` and the
    inverse noise matrix ``P(y|labels)`` from the `confident_joint` ``count(labels, y)``. The
    `confident_joint` can be estimated by `~cleanlab.count.compute_confident_joint`
    which counts confident examples.

    Parameters
    ----------
    confident_joint : np.ndarray
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    labels : np.ndarray
      A 1D array of shape ``(N,)`` containing class labels for a standard (multi-class) classification dataset. Some given labels may be erroneous.
      Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.

    py_method : {"cnt", "eqn", "marginal", "marginal_ps"}, default="cnt"
      `py` is shorthand for the "class proportions (a.k.a prior) of the true labels".
      This method defines how to compute the latent prior ``p(true_label=k)``. Default is ``"cnt"``,
      which works well even when the noise matrices are estimated poorly by using
      the matrix diagonals instead of all the probabilities.

    converge_latent_estimates : bool, optional
      If ``True``, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    Returns
    ------
    tuple
      A tuple containing (py, noise_matrix, inv_noise_matrix).

    Note
    ----
    Multi-label classification is not supported in this method.
    """

    num_classes = len(confident_joint)
    label_counts = value_counts_fill_missing_classes(labels, num_classes)
    # 'ps' is p(labels=k)
    ps = label_counts / float(len(labels))
    # Number of training examples confidently counted from each noisy class
    labels_class_counts = confident_joint.sum(axis=1).astype(float)
    # Number of training examples confidently counted into each true class
    true_labels_class_counts = confident_joint.sum(axis=0).astype(float)
    # p(label=k_s|true_label=k_y) ~ |label=k_s and true_label=k_y| / |true_label=k_y|
    noise_matrix = confident_joint / np.clip(true_labels_class_counts, a_min=TINY_VALUE, a_max=None)
    # p(true_label=k_y|label=k_s) ~ |true_label=k_y and label=k_s| / |label=k_s|
    inv_noise_matrix = confident_joint.T / np.clip(
        labels_class_counts, a_min=TINY_VALUE, a_max=None
    )
    # Compute the prior p(y), the latent (uncorrupted) class distribution.
    py = compute_py(
        ps,
        noise_matrix,
        inv_noise_matrix,
        py_method=py_method,
        true_labels_class_counts=true_labels_class_counts,
    )
    # Clip noise rates to be valid probabilities.
    noise_matrix = clip_noise_rates(noise_matrix)
    inv_noise_matrix = clip_noise_rates(inv_noise_matrix)
    # Make latent estimates mathematically agree in their algebraic relations.
    if converge_latent_estimates:
        py, noise_matrix, inv_noise_matrix = _converge_estimates(
            ps, py, noise_matrix, inv_noise_matrix
        )
        # Again clip py and noise rates into proper range [0,1)
        py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
        noise_matrix = clip_noise_rates(noise_matrix)
        inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    return py, noise_matrix, inv_noise_matrix

def get_confident_thresholds(
    labels: LabelLike,
    pred_probs: np.ndarray,
    multi_label: bool = False,
) -> np.ndarray:
    """Returns expected (average) "self-confidence" for each class.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j,
    i.e. the model-predicted probability of this class averaged amongst all examples labeled as class j.

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    multi_label : bool, default = False
      Set ``False`` if your dataset is for regular (multi-class) classification, where each example belongs to exactly one class.
      Set ``True`` if your dataset is for multi-label classification, where each example can belong to multiple classes.
      See documentation of `~cleanlab.count.compute_confident_joint` for details.

    Returns
    -------
    confident_thresholds : np.ndarray
      An array of shape ``(K, )`` where K is the number of classes.
    """
    if multi_label:
        assert isinstance(labels, list)
        return _get_confident_thresholds_multilabel(labels=labels, pred_probs=pred_probs)
    else:
        # When all_classes != unique_classes the class threshold for the missing classes is set to
        # BIG_VALUE such that no valid prob >= BIG_VALUE (no example will be counted in missing classes)
        # REQUIRES: pred_probs.max() >= 1
        # TODO: if you want this to work for arbitrary softmax outputs where pred_probs.max()
        #  may exceed 1, change BIG_VALUE = 2 --> BIG_VALUE = 2 * pred_probs.max(). Downside of
        #  this approach is that there will be no standard value returned for missing classes.
        labels = labels_to_array(labels)
        all_classes = range(pred_probs.shape[1])
        unique_classes = get_unique_classes(labels, multi_label=multi_label)
        BIG_VALUE = 2
        confident_thresholds = [
            np.mean(pred_probs[:, k][labels == k]) if k in unique_classes else BIG_VALUE
            for k in all_classes
        ]
        confident_thresholds = np.clip(
            confident_thresholds, a_min=CONFIDENT_THRESHOLDS_LOWER_BOUND, a_max=None
        )
        return confident_thresholds

