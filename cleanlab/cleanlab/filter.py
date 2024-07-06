def find_label_issues(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs: Optional[Dict[str, Any]] = None,
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    multi_label: bool = False,
) -> np.ndarray:
    """
    Identifies potentially bad labels in a classification dataset using confident learning.

    Returns a boolean mask for the entire dataset where ``True`` represents
    an example identified with a label issue and ``False`` represents an example that seems correctly labeled.

    Instead of a mask, you can obtain indices of the examples with label issues in your dataset
    (sorted by issue severity) by specifying the `return_indices_ranked_by` argument.
    This determines which label quality score is used to quantify severity,
    and is useful to view only the top-`J` most severe issues in your dataset.

    The number of indices returned as issues is controlled by `frac_noise`: reduce its
    value to identify fewer label issues. If you aren't sure, leave this set to 1.0.

    Tip: if you encounter the error "pred_probs is not defined", try setting
    ``n_jobs=1``.

    Parameters
    ----------
    labels : np.ndarray or list
      A discrete vector of noisy labels for a classification dataset, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, each label must be integer in 0, 1, ..., K-1.
      For a standard (multi-class) classification dataset where each example is labeled with one class,
      `labels` should be 1D array of shape ``(N,)``, for example: ``labels = [1,0,2,1,1,0...]``.

    pred_probs : np.ndarray, optional
      An array of shape ``(N, K)`` of model-predicted class probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1.

      **Note**: Returned label issues are most accurate when they are computed based on out-of-sample `pred_probs` from your model.
      To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      This is encouraged to get better results.

    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default=None
      Determines what is returned by this method: either a boolean mask or list of indices np.ndarray.
      If ``None``, this function returns a boolean mask (``True`` if example at index is label error).
      If not ``None``, this function returns a sorted array of indices of examples with label issues
      (instead of a boolean mask). Indices are sorted by label quality score which can be one of:

      - ``'normalized_margin'``: ``normalized margin (p(label = k) - max(p(label != k)))``
      - ``'self_confidence'``: ``[pred_probs[i][labels[i]] for i in label_issues_idx]``
      - ``'confidence_weighted_entropy'``: ``entropy(pred_probs) / self_confidence``

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into scoring functions for ranking by
      label quality score (see :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`).

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given', 'low_normalized_margin', 'low_self_confidence'}, default='prune_by_noise_rate'
      Method to determine which examples are flagged as having label issue, so you can filter/prune them from the dataset. Options:

      - ``'prune_by_noise_rate'``: filters examples with *high probability* of being mislabeled for every non-diagonal in the confident joint (see `prune_counts_matrix` in `filter.py`). These are the examples where (with high confidence) the given label is unlikely to match the predicted label for the example.
      - ``'prune_by_class'``: filters the examples with *smallest probability* of belonging to their given class label for every class.
      - ``'both'``: filters only those examples that would be filtered by both ``'prune_by_noise_rate'`` and ``'prune_by_class'``.
      - ``'confident_learning'``: filters the examples counted as part of the off-diagonals of the confident joint. These are the examples that are confidently predicted to be a different label than their given label.
      - ``'predicted_neq_given'``: filters examples for which the predicted class (i.e. argmax of the predicted probabilities) does not match the given label.
      - ``'low_normalized_margin'``: filters the examples with *smallest* normalized margin label quality score. The number of issues returned matches :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`.
      - ``'low_self_confidence'``: filters the examples with *smallest* self confidence label quality score. The number of issues returned matches :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`.

    frac_noise : float, default=1.0
      Used to only return the "top" ``frac_noise * num_label_issues``. The choice of which "top"
      label issues to return is dependent on the `filter_by` method used. It works by reducing the
      size of the off-diagonals of the `joint` distribution of given labels and true labels
      proportionally by `frac_noise` prior to estimating label issues with each method.
      This parameter only applies for `filter_by=both`, `filter_by=prune_by_class`, and
      `filter_by=prune_by_noise_rate` methods and currently is unused by other methods.
      When ``frac_noise=1.0``, return all "confident" estimated noise indices (recommended).

      frac_noise * number_of_mislabeled_examples_in_class_k.

    num_to_remove_per_class : array_like
      An iterable of length K, the number of classes.
      E.g. if K = 3, ``num_to_remove_per_class=[5, 0, 1]`` would return
      the indices of the 5 most likely mislabeled examples in class 0,
      and the most likely mislabeled example in class 2.

      Note
      ----
      Only set this parameter if ``filter_by='prune_by_class'``.
      You may use with ``filter_by='prune_by_noise_rate'``, but
      if ``num_to_remove_per_class=k``, then either k-1, k, or k+1
      examples may be removed for any class due to rounding error. If you need
      exactly 'k' examples removed from every class, you should use
      ``filter_by='prune_by_class'``.

    min_examples_per_class : int, default=1
      Minimum number of examples per class to avoid flagging as label issues.
      This is useful to avoid deleting too much data from one class
      when pruning noisy examples in datasets with rare classes.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    n_jobs : optional
      Number of processing threads used by multiprocessing. Default ``None``
      sets to the number of cores on your CPU (physical cores if you have ``psutil`` package installed, otherwise logical cores).
      Set this to 1 to *disable* parallel processing (if its causing issues).
      Windows users may see a speed-up with ``n_jobs=1``.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    Returns
    -------
    label_issues : np.ndarray
      If `return_indices_ranked_by` left unspecified, returns a boolean **mask** for the entire dataset
      where ``True`` represents a label issue and ``False`` represents an example that is
      accurately labeled with high confidence.
      If `return_indices_ranked_by` is specified, returns a shorter array of **indices** of examples identified to have
      label issues (i.e. those indices where the mask would be ``True``), sorted by likelihood that the corresponding label is correct.

      Note
      ----
      Obtain the *indices* of examples with label issues in your dataset by setting `return_indices_ranked_by`.
    """
    if not rank_by_kwargs:
        rank_by_kwargs = {}

    assert filter_by in [
        "low_normalized_margin",
        "low_self_confidence",
        "prune_by_noise_rate",
        "prune_by_class",
        "both",
        "confident_learning",
        "predicted_neq_given",
    ]  # TODO: change default to confident_learning ?
    allow_one_class = False
    if isinstance(labels, np.ndarray) or all(isinstance(lab, int) for lab in labels):
        if set(labels) == {0}:  # occurs with missing classes in multi-label settings
            allow_one_class = True
    assert_valid_inputs(
        X=None,
        y=labels,
        pred_probs=pred_probs,
        multi_label=multi_label,
        allow_one_class=allow_one_class,
    )

    if filter_by in [
        "confident_learning",
        "predicted_neq_given",
        "low_normalized_margin",
        "low_self_confidence",
    ] and (frac_noise != 1.0 or num_to_remove_per_class is not None):
        warn_str = (
            "frac_noise and num_to_remove_per_class parameters are only supported"
            " for filter_by 'prune_by_noise_rate', 'prune_by_class', and 'both'. They "
            "are not supported for methods 'confident_learning', 'predicted_neq_given', "
            "'low_normalized_margin' or 'low_self_confidence'."
        )
        warnings.warn(warn_str)
    if (num_to_remove_per_class is not None) and (
        filter_by
        in [
            "confident_learning",
            "predicted_neq_given",
            "low_normalized_margin",
            "low_self_confidence",
        ]
    ):
        # TODO - add support for these filters
        raise ValueError(
            "filter_by 'confident_learning', 'predicted_neq_given', 'low_normalized_margin' "
            "or 'low_self_confidence' is not supported (yet) when setting 'num_to_remove_per_class'"
        )
    if filter_by == "confident_learning" and isinstance(confident_joint, np.ndarray):
        warn_str = (
            "The supplied `confident_joint` is ignored when `filter_by = 'confident_learning'`; confident joint will be "
            "re-estimated from the given labels. To use your supplied `confident_joint`, please specify a different "
            "`filter_by` value."
        )
        warnings.warn(warn_str)

    K = get_num_classes(
        labels=labels, pred_probs=pred_probs, label_matrix=confident_joint, multi_label=multi_label
    )
    # Boolean set to true if dataset is large
    big_dataset = K * len(labels) > 1e8

    # Set-up number of multiprocessing threads
    # On Windows/macOS, when multi_label is True, multiprocessing is much slower
    # even for faily large input arrays, so we default to n_jobs=1 in this case
    os_name = platform.system()
    if n_jobs is None:
        if multi_label and os_name != "Linux":
            n_jobs = 1
        else:
            if psutil_exists:
                n_jobs = psutil.cpu_count(logical=False)  # physical cores
            elif big_dataset:
                print(
                    "To default `n_jobs` to the number of physical cores for multiprocessing in find_label_issues(), please: `pip install psutil`.\n"
                    "Note: You can safely ignore this message. `n_jobs` only affects runtimes, results will be the same no matter its value.\n"
                    "Since psutil is not installed, `n_jobs` was set to the number of logical cores by default.\n"
                    "Disable this message by either installing psutil or specifying the `n_jobs` argument."
                )  # pragma: no cover
            if not n_jobs:
                # either psutil does not exist
                # or psutil can return None when physical cores cannot be determined
                # switch to logical cores
                n_jobs = multiprocessing.cpu_count()
    else:
        assert n_jobs >= 1

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        warnings.warn(
            "The multi_label argument to filter.find_label_issues() is deprecated and will be removed in future versions. Please use `multilabel_classification.filter.find_label_issues()` instead.",
            DeprecationWarning,
        )
        return _find_label_issues_multilabel(
            labels,
            pred_probs,
            return_indices_ranked_by,
            rank_by_kwargs,
            filter_by,
            frac_noise,
            num_to_remove_per_class,
            min_examples_per_class,
            confident_joint,
            n_jobs,
            verbose,
        )

    # Else this is standard multi-class classification
    # Number of examples in each class of labels
    label_counts = value_counts_fill_missing_classes(labels, K, multi_label=multi_label)
    # Ensure labels are of type np.ndarray()
    labels = np.asarray(labels)
    if confident_joint is None or filter_by == "confident_learning":
        from cleanlab.count import compute_confident_joint

        confident_joint, cl_error_indices = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            multi_label=multi_label,
            return_indices_of_off_diagonals=True,
        )

    if filter_by in ["low_normalized_margin", "low_self_confidence"]:
        # TODO: consider setting adjust_pred_probs to true based on benchmarks (or adding it kwargs, or ignoring and leaving as false by default)
        scores = get_label_quality_scores(
            labels,
            pred_probs,
            method=filter_by[4:],
            adjust_pred_probs=False,
        )
        num_errors = num_label_issues(
            labels, pred_probs, multi_label=multi_label  # TODO: Check usage of multilabel
        )
        # Find label issues O(nlogn) solution (mapped to boolean mask later in the method)
        cl_error_indices = np.argsort(scores)[:num_errors]
        # The following is the O(n) fastest solution (check for one-off errors), but the problem is if lots of the scores are identical you will overcount,
        # you can end up returning more or less and they aren't ranked in the boolean form so there's no way to drop the highest scores randomly
        #     boundary = np.partition(scores, num_errors)[num_errors]  # O(n) solution
        #     label_issues_mask = scores <= boundary

    if filter_by in ["prune_by_noise_rate", "prune_by_class", "both"]:
        # Create `prune_count_matrix` with the number of examples to remove in each class and
        # leave at least min_examples_per_class examples per class.
        # `prune_count_matrix` is transposed relative to the confident_joint.
        prune_count_matrix = _keep_at_least_n_per_class(
            prune_count_matrix=confident_joint.T,
            n=min_examples_per_class,
            frac_noise=frac_noise,
        )

        if num_to_remove_per_class is not None:
            # Estimate joint probability distribution over label issues
            psy = prune_count_matrix / np.sum(prune_count_matrix, axis=1)
            noise_per_s = psy.sum(axis=1) - psy.diagonal()
            # Calibrate labels.t. noise rates sum to num_to_remove_per_class
            tmp = (psy.T * num_to_remove_per_class / noise_per_s).T
            np.fill_diagonal(tmp, label_counts - num_to_remove_per_class)
            prune_count_matrix = round_preserving_row_totals(tmp)

        # Prepare multiprocessing shared data
        # On Linux, multiprocessing is started with fork,
        # so data can be shared with global vairables + COW
        # On Window/macOS, processes are started with spawn,
        # so data will need to be pickled to the subprocesses through input args
        chunksize = max(1, K // n_jobs)
        if n_jobs == 1 or os_name == "Linux":
            global pred_probs_by_class, prune_count_matrix_cols
            pred_probs_by_class = {k: pred_probs[labels == k] for k in range(K)}
            prune_count_matrix_cols = {k: prune_count_matrix[:, k] for k in range(K)}
            args = [[k, min_examples_per_class, None] for k in range(K)]
        else:
            args = [
                [k, min_examples_per_class, [pred_probs[labels == k], prune_count_matrix[:, k]]]
                for k in range(K)
            ]

    # Perform Pruning with threshold probabilities from BFPRT algorithm in O(n)
    # Operations are parallelized across all CPU processes
    if filter_by == "prune_by_class" or filter_by == "both":
        if n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as p:
                if verbose:  # pragma: no cover
                    print("Parallel processing label issues by class.")
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    label_issues_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_class, args, chunksize=chunksize), total=K)
                    )
                else:
                    label_issues_masks_per_class = p.map(_prune_by_class, args, chunksize=chunksize)
        else:
            label_issues_masks_per_class = [_prune_by_class(arg) for arg in args]

        label_issues_mask = np.zeros(len(labels), dtype=bool)
        for k, mask in enumerate(label_issues_masks_per_class):
            if len(mask) > 1:
                label_issues_mask[labels == k] = mask

    if filter_by == "both":
        label_issues_mask_by_class = label_issues_mask

    if filter_by == "prune_by_noise_rate" or filter_by == "both":
        if n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as p:
                if verbose:  # pragma: no cover
                    print("Parallel processing label issues by noise rate.")
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    label_issues_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_count, args, chunksize=chunksize), total=K)
                    )
                else:
                    label_issues_masks_per_class = p.map(_prune_by_count, args, chunksize=chunksize)
        else:
            label_issues_masks_per_class = [_prune_by_count(arg) for arg in args]

        label_issues_mask = np.zeros(len(labels), dtype=bool)
        for k, mask in enumerate(label_issues_masks_per_class):
            if len(mask) > 1:
                label_issues_mask[labels == k] = mask

    if filter_by == "both":
        label_issues_mask = label_issues_mask & label_issues_mask_by_class

    if filter_by in ["confident_learning", "low_normalized_margin", "low_self_confidence"]:
        label_issues_mask = np.zeros(len(labels), dtype=bool)
        label_issues_mask[cl_error_indices] = True

    if filter_by == "predicted_neq_given":
        label_issues_mask = find_predicted_neq_given(labels, pred_probs, multi_label=multi_label)

    if filter_by not in ["low_self_confidence", "low_normalized_margin"]:
        # Remove label issues if model prediction is close to given label
        mask = _reduce_issues(pred_probs=pred_probs, labels=labels)
        label_issues_mask[mask] = False

    if verbose:
        print("Number of label issues found: {}".format(sum(label_issues_mask)))

    # TODO: run count.num_label_issues() and adjust the total issues found here to match
    if return_indices_ranked_by is not None:
        er = order_label_issues(
            label_issues_mask=label_issues_mask,
            labels=labels,
            pred_probs=pred_probs,
            rank_by=return_indices_ranked_by,
            rank_by_kwargs=rank_by_kwargs,
        )
        return er
    return label_issues_mask

def find_predicted_neq_given(
    labels: LabelLike, pred_probs: np.ndarray, *, multi_label: bool = False
) -> np.ndarray:
    """A simple baseline approach that considers ``argmax(pred_probs) != labels`` as the examples with label issues.

    Parameters
    ----------
    labels : np.ndarray or list
      Labels in the same format expected by the `~cleanlab.filter.find_label_issues` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.filter.find_label_issues` function.

    multi_label : bool, optional
      Whether each example may have multiple labels or not (see documentation for the `~cleanlab.filter.find_label_issues` function).

    Returns
    -------
    label_issues_mask : np.ndarray
      A boolean mask for the entire dataset where ``True`` represents a
      label issue and ``False`` represents an example that is accurately
      labeled with high confidence.
    """

    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=multi_label)
    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _find_predicted_neq_given_multilabel(labels=labels, pred_probs=pred_probs)
    else:
        return np.argmax(pred_probs, axis=1) != np.asarray(labels)

