def get_label_quality_multiannotator(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: np.ndarray,
    *,
    consensus_method: Union[str, List[str]] = "best_quality",
    quality_method: str = "crowdlab",
    calibrate_probs: bool = False,
    return_detailed_quality: bool = True,
    return_annotator_stats: bool = True,
    return_weights: bool = False,
    verbose: bool = True,
    label_quality_score_kwargs: dict = {},
) -> Dict[str, Any]:
    """Returns label quality scores for each example and for each annotator in a dataset labeled by multiple annotators.

    This function is for multiclass classification datasets where examples have been labeled by
    multiple annotators (not necessarily the same number of annotators per example).

    It computes one consensus label for each example that best accounts for the labels chosen by each
    annotator (and their quality), as well as a consensus quality score for how confident we are that this consensus label is actually correct.
    It also computes similar quality scores for each annotator's individual labels, and the quality of each annotator.
    Scores are between 0 and 1 (estimated via methods like CROWDLAB); lower scores indicate labels/annotators less likely to be correct.

    To decide what data to collect additional labels for, try the `~cleanlab.multiannotator.get_active_learning_scores`
    (ActiveLab) function, which is intended for active learning with multiple annotators.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        ``labels_multiannotator[n][m]`` = label for n-th example given by m-th annotator.

        For a dataset with K classes, each given label must be an integer in 0, 1, ..., K-1 or ``NaN`` if this annotator did not label a particular example.
        If you have string or other differently formatted labels, you can convert them to the proper format using :py:func:`format_multiannotator_labels <cleanlab.internal.multiannotator_utils.format_multiannotator_labels>`.
        If pd.DataFrame, column names should correspond to each annotator's ID.
    pred_probs : np.ndarray
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
    consensus_method : str or List[str], default = "majority_vote"
        Specifies the method used to aggregate labels from multiple annotators into a single consensus label.
        Options include:

        * ``majority_vote``: consensus obtained using a simple majority vote among annotators, with ties broken via ``pred_probs``.
        * ``best_quality``: consensus obtained by selecting the label with highest label quality (quality determined by method specified in ``quality_method``).

        A List may be passed if you want to consider multiple methods for producing consensus labels.
        If a List is passed, then the 0th element of the list is the method used to produce columns `consensus_label`, `consensus_quality_score`, `annotator_agreement` in the returned DataFrame.
        The remaning (1st, 2nd, 3rd, etc.) elements of this list are output as extra columns in the returned pandas DataFrame with names formatted as:
        `consensus_label_SUFFIX`, `consensus_quality_score_SUFFIX` where `SUFFIX` = each element of this
        list, which must correspond to a valid method for computing consensus labels.
    quality_method : str, default = "crowdlab"
        Specifies the method used to calculate the quality of the consensus label.
        Options include:

        * ``crowdlab``: an emsemble method that weighs both the annotators' labels as well as the model's prediction.
        * ``agreement``: the fraction of annotators that agree with the consensus label.
    calibrate_probs : bool, default = False
        Boolean value that specifies whether the provided `pred_probs` should be re-calibrated to better match the annotators' empirical label distribution.
        We recommend setting this to True in active learning applications, in order to prevent overconfident models from suggesting the wrong examples to collect labels for.
    return_detailed_quality: bool, default = True
        Boolean to specify if `detailed_label_quality` is returned.
    return_annotator_stats : bool, default = True
        Boolean to specify if `annotator_stats` is returned.
    return_weights : bool, default = False
        Boolean to specify if `model_weight` and `annotator_weight` is returned.
        Model and annotator weights are applicable for ``quality_method == crowdlab``, will return ``None`` for any other quality methods.
    verbose : bool, default = True
        Important warnings and other printed statements may be suppressed if ``verbose`` is set to ``False``.
    label_quality_score_kwargs : dict, optional
        Keyword arguments to pass into :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    Returns
    -------
    labels_info : dict
        Dictionary containing up to 5 pandas DataFrame with keys as below:

        ``label_quality`` : pandas.DataFrame
            pandas DataFrame in which each row corresponds to one example, with columns:

            * ``num_annotations``: the number of annotators that have labeled each example.
            * ``consensus_label``: the single label that is best for each example (you can control how it is derived from all annotators' labels via the argument: ``consensus_method``).
            * ``annotator_agreement``: the fraction of annotators that agree with the consensus label (only consider the annotators that labeled that particular example).
            * ``consensus_quality_score``: label quality score for consensus label, calculated by the method specified in ``quality_method``.

        ``detailed_label_quality`` : pandas.DataFrame
            Only returned if `return_detailed_quality=True`.
            Returns a pandas DataFrame with columns `quality_annotator_1`, `quality_annotator_2`, ..., `quality_annotator_M` where each entry is
            the label quality score for the labels provided by each annotator (is ``NaN`` for examples which this annotator did not label).

        ``annotator_stats`` : pandas.DataFrame
            Only returned if `return_annotator_stats=True`.
            Returns overall statistics about each annotator, sorted by lowest annotator_quality first.
            pandas DataFrame in which each row corresponds to one annotator (the row IDs correspond to annotator IDs), with columns:

            * ``annotator_quality``: overall quality of a given annotator's labels, calculated by the method specified in ``quality_method``.
            * ``num_examples_labeled``: number of examples annotated by a given annotator.
            * ``agreement_with_consensus``: fraction of examples where a given annotator agrees with the consensus label.
            * ``worst_class``: the class that is most frequently mislabeled by a given annotator.

        ``model_weight`` : float
            Only returned if `return_weights=True`. It is only applicable for ``quality_method == crowdlab``.
            The model weight specifies the weight of classifier model in weighted averages used to estimate label quality
            This number is an estimate of how trustworthy the model is relative the annotators.

        ``annotator_weight`` : np.ndarray
            Only returned if `return_weights=True`. It is only applicable for ``quality_method == crowdlab``.
            An array of shape ``(M,)`` where M is the number of annotators, specifying the weight of each annotator in weighted averages used to estimate label quality.
            These weights are estimates of how trustworthy each annotator is relative to the other annotators.

    """

    if isinstance(labels_multiannotator, pd.DataFrame):
        annotator_ids = labels_multiannotator.columns
        index_col = labels_multiannotator.index
        labels_multiannotator = (
            labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
        )
    elif isinstance(labels_multiannotator, np.ndarray):
        annotator_ids = None
        index_col = None
    else:
        raise ValueError("labels_multiannotator must be either a NumPy array or Pandas DataFrame.")

    if return_weights == True and quality_method != "crowdlab":
        raise ValueError(
            "Model and annotator weights are only applicable to the crowdlab quality method. "
            "Either set return_weights=False or quality_method='crowdlab'."
        )

    assert_valid_inputs_multiannotator(
        labels_multiannotator, pred_probs, annotator_ids=annotator_ids
    )

    # Count number of non-NaN values for each example
    num_annotations = np.sum(~np.isnan(labels_multiannotator), axis=1)

    # calibrate pred_probs
    if calibrate_probs:
        optimal_temp = find_best_temp_scaler(labels_multiannotator, pred_probs)
        pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)

    if not isinstance(consensus_method, list):
        consensus_method = [consensus_method]

    if "best_quality" in consensus_method or "majority_vote" in consensus_method:
        majority_vote_label = get_majority_vote_label(
            labels_multiannotator=labels_multiannotator,
            pred_probs=pred_probs,
            verbose=False,
        )
        (
            MV_annotator_agreement,
            MV_consensus_quality_score,
            MV_post_pred_probs,
            MV_model_weight,
            MV_annotator_weight,
        ) = _get_consensus_stats(
            labels_multiannotator=labels_multiannotator,
            pred_probs=pred_probs,
            num_annotations=num_annotations,
            consensus_label=majority_vote_label,
            quality_method=quality_method,
            verbose=verbose,
            label_quality_score_kwargs=label_quality_score_kwargs,
        )

    label_quality = pd.DataFrame({"num_annotations": num_annotations}, index=index_col)
    valid_methods = ["majority_vote", "best_quality"]
    main_method = True

    for curr_method in consensus_method:
        # geting consensus label and stats
        if curr_method == "majority_vote":
            consensus_label = majority_vote_label
            annotator_agreement = MV_annotator_agreement
            consensus_quality_score = MV_consensus_quality_score
            post_pred_probs = MV_post_pred_probs
            model_weight = MV_model_weight
            annotator_weight = MV_annotator_weight

        elif curr_method == "best_quality":
            consensus_label = np.full(len(majority_vote_label), np.nan)
            for i in range(len(consensus_label)):
                max_pred_probs_ind = np.where(
                    MV_post_pred_probs[i] == np.max(MV_post_pred_probs[i])
                )[0]
                if len(max_pred_probs_ind) == 1:
                    consensus_label[i] = max_pred_probs_ind[0]
                else:
                    consensus_label[i] = majority_vote_label[i]
            consensus_label = consensus_label.astype(int)  # convert all label types to int

            (
                annotator_agreement,
                consensus_quality_score,
                post_pred_probs,
                model_weight,
                annotator_weight,
            ) = _get_consensus_stats(
                labels_multiannotator=labels_multiannotator,
                pred_probs=pred_probs,
                num_annotations=num_annotations,
                consensus_label=consensus_label,
                quality_method=quality_method,
                verbose=verbose,
                label_quality_score_kwargs=label_quality_score_kwargs,
            )

        else:
            raise ValueError(
                f"""
                {curr_method} is not a valid consensus method!
                Please choose a valid consensus_method: {valid_methods}
                """
            )

        if verbose:
            # check if any classes no longer appear in the set of consensus labels
            check_consensus_label_classes(
                labels_multiannotator=labels_multiannotator,
                consensus_label=consensus_label,
                consensus_method=curr_method,
            )

        # saving stats into dataframe, computing additional stats if specified
        if main_method:
            (
                label_quality["consensus_label"],
                label_quality["consensus_quality_score"],
                label_quality["annotator_agreement"],
            ) = (
                consensus_label,
                consensus_quality_score,
                annotator_agreement,
            )

            label_quality = label_quality.reindex(
                columns=[
                    "consensus_label",
                    "consensus_quality_score",
                    "annotator_agreement",
                    "num_annotations",
                ]
            )

            # default variable for _get_annotator_stats
            detailed_label_quality = None

            if return_detailed_quality:
                # Compute the label quality scores for each annotators' labels
                detailed_label_quality = np.apply_along_axis(
                    _get_annotator_label_quality_score,
                    axis=0,
                    arr=labels_multiannotator,
                    pred_probs=post_pred_probs,
                    label_quality_score_kwargs=label_quality_score_kwargs,
                )
                detailed_label_quality_df = pd.DataFrame(
                    detailed_label_quality, index=index_col, columns=annotator_ids
                ).add_prefix("quality_annotator_")

            if return_annotator_stats:
                annotator_stats = _get_annotator_stats(
                    labels_multiannotator=labels_multiannotator,
                    pred_probs=post_pred_probs,
                    consensus_label=consensus_label,
                    num_annotations=num_annotations,
                    annotator_agreement=annotator_agreement,
                    model_weight=model_weight,
                    annotator_weight=annotator_weight,
                    consensus_quality_score=consensus_quality_score,
                    detailed_label_quality=detailed_label_quality,
                    annotator_ids=annotator_ids,
                    quality_method=quality_method,
                )

            main_method = False

        else:
            (
                label_quality[f"consensus_label_{curr_method}"],
                label_quality[f"consensus_quality_score_{curr_method}"],
                label_quality[f"annotator_agreement_{curr_method}"],
            ) = (
                consensus_label,
                consensus_quality_score,
                annotator_agreement,
            )

    labels_info = {
        "label_quality": label_quality,
    }

    if return_detailed_quality:
        labels_info["detailed_label_quality"] = detailed_label_quality_df
    if return_annotator_stats:
        labels_info["annotator_stats"] = annotator_stats
    if return_weights:
        labels_info["model_weight"] = model_weight
        labels_info["annotator_weight"] = annotator_weight

    return labels_info

def get_active_learning_scores(
    labels_multiannotator: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    pred_probs: Optional[np.ndarray] = None,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an ActiveLab quality score for each example in the dataset, to estimate which examples are most informative to (re)label next in active learning.

    We consider settings where one example can be labeled by one or more annotators and some examples have no labels at all so far.

    The score is in between 0 and 1, and can be used to prioritize what data to collect additional labels for.
    Lower scores indicate examples whose true label we are least confident about based on the current data;
    collecting additional labels for these low-scoring examples will be more informative than collecting labels for other examples.
    To use an annotation budget most efficiently, select a batch of examples with the lowest scores and collect one additional label for each example,
    and repeat this process after retraining your classifier.

    You can use this function to get active learning scores for: examples that already have one or more labels (specify ``labels_multiannotator`` and ``pred_probs``
    as arguments), or for unlabeled examples (specify ``pred_probs_unlabeled``), or for both types of examples (specify all of the above arguments).

    To analyze a fixed dataset labeled by multiple annotators rather than collecting additional labels, try the
    `~cleanlab.multiannotator.get_label_quality_multiannotator` (CROWDLAB) function instead.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray, optional
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators. Note that this function also works with
        datasets where there is only one annotator (M=1).
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
        Note that examples that have no annotator labels should not be included in this DataFrame/array.
        This argument is optional if ``pred_probs`` is not provided (you might only provide ``pred_probs_unlabeled`` to only get active learning scores for the unlabeled examples).
    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
        This argument is optional if you only want to get active learning scores for unlabeled examples (specify only ``pred_probs_unlabeled`` instead).
    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model for examples that have no annotator labels.
        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
        This argument is optional if you only want to get active learning scores for already-labeled examples (specify only ``pred_probs`` instead).

    Returns
    -------
    active_learning_scores : np.ndarray
        Array of shape ``(N,)`` indicating the ActiveLab quality scores for each example.
        This array is empty if no already-labeled data was provided via ``labels_multiannotator``.
        Examples with the lowest scores are those we should label next in order to maximally improve our classifier model.

    active_learning_scores_unlabeled : np.ndarray
        Array of shape ``(N,)`` indicating the active learning quality scores for each unlabeled example.
        Returns an empty array if no unlabeled data is provided.
        Examples with the lowest scores are those we should label next in order to maximally improve our classifier model
        (scores for unlabeled data are directly comparable with the `active_learning_scores` for labeled data).
    """

    assert_valid_pred_probs(pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled)

    # compute multiannotator stats if labeled data is provided
    if pred_probs is not None:
        if labels_multiannotator is None:
            raise ValueError(
                "labels_multiannotator cannot be None when passing in pred_probs. ",
                "Either provide labels_multiannotator to obtain active learning scores for the labeled examples, "
                "or just pass in pred_probs_unlabeled to get active learning scores for unlabeled examples.",
            )

        if isinstance(labels_multiannotator, pd.DataFrame):
            labels_multiannotator = (
                labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
            )
        elif not isinstance(labels_multiannotator, np.ndarray):
            raise ValueError(
                "labels_multiannotator must be either a NumPy array or Pandas DataFrame."
            )
        # check that labels_multiannotator is a 2D array
        if labels_multiannotator.ndim != 2:
            raise ValueError(
                "labels_multiannotator must be a 2D array or dataframe, "
                "each row represents an example and each column represents an annotator."
            )

        num_classes = get_num_classes(pred_probs=pred_probs)

        # if all examples are only labeled by a single annotator
        if (np.sum(~np.isnan(labels_multiannotator), axis=1) == 1).all():
            optimal_temp = 1.0  # do not temp scale for single annotator case, temperature is defined here for later use

            assert_valid_inputs_multiannotator(
                labels_multiannotator, pred_probs, allow_single_label=True
            )

            consensus_label = get_majority_vote_label(
                labels_multiannotator=labels_multiannotator,
                pred_probs=pred_probs,
                verbose=False,
            )
            quality_of_consensus_labeled = get_label_quality_scores(consensus_label, pred_probs)
            model_weight = 1
            annotator_weight = np.full(labels_multiannotator.shape[1], 1)
            avg_annotator_weight = np.mean(annotator_weight)

        # examples are annotated by multiple annotators
        else:
            optimal_temp = find_best_temp_scaler(labels_multiannotator, pred_probs)
            pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)

            multiannotator_info = get_label_quality_multiannotator(
                labels_multiannotator,
                pred_probs,
                return_annotator_stats=False,
                return_detailed_quality=False,
                return_weights=True,
            )

            quality_of_consensus_labeled = multiannotator_info["label_quality"][
                "consensus_quality_score"
            ]
            model_weight = multiannotator_info["model_weight"]
            annotator_weight = multiannotator_info["annotator_weight"]
            avg_annotator_weight = np.mean(annotator_weight)

        # compute scores for labeled data
        active_learning_scores = np.full(len(labels_multiannotator), np.nan)
        for i, annotator_labels in enumerate(labels_multiannotator):
            active_learning_scores[i] = np.average(
                (quality_of_consensus_labeled[i], 1 / num_classes),
                weights=(
                    np.sum(annotator_weight[~np.isnan(annotator_labels)]) + model_weight,
                    avg_annotator_weight,
                ),
            )

    # no labeled data provided so do not estimate temperature and model/annotator weights
    elif pred_probs_unlabeled is not None:
        num_classes = get_num_classes(pred_probs=pred_probs_unlabeled)
        optimal_temp = 1
        model_weight = 1
        avg_annotator_weight = 1
        active_learning_scores = np.array([])

    else:
        raise ValueError(
            "pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two."
        )

    # compute scores for unlabeled data
    if pred_probs_unlabeled is not None:
        pred_probs_unlabeled = temp_scale_pred_probs(pred_probs_unlabeled, optimal_temp)
        quality_of_consensus_unlabeled = np.max(pred_probs_unlabeled, axis=1)

        active_learning_scores_unlabeled = np.average(
            np.stack(
                [
                    quality_of_consensus_unlabeled,
                    np.full(len(quality_of_consensus_unlabeled), 1 / num_classes),
                ]
            ),
            weights=[model_weight, avg_annotator_weight],
            axis=0,
        )

    else:
        active_learning_scores_unlabeled = np.array([])

    return active_learning_scores, active_learning_scores_unlabeled

def get_active_learning_scores_ensemble(
    labels_multiannotator: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    pred_probs: Optional[np.ndarray] = None,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an ActiveLab quality score for each example in the dataset, based on predictions from an ensemble of models.

    This function is similar to `~cleanlab.multiannotator.get_active_learning_scores` but allows for an
    ensemble of multiple classifier models to be trained and will aggregate predictions from the models to compute the ActiveLab quality score.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        Multiannotator labels in the same format expected by `~cleanlab.multiannotator.get_active_learning_scores`.
        This argument is optional if ``pred_probs`` is not provided (in cases where you only provide ``pred_probs_unlabeled`` to get active learning scores for unlabeled examples).
    pred_probs : np.ndarray
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.
        Note that this function also works with datasets where there is only one annotator (M=1).
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
        This argument is optional if you only want to get active learning scores for unlabeled examples (pass in ``pred_probs_unlabeled`` instead).
    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from a trained classifier model
        for examples that have no annotated labels so far (but which we may want to label in the future, and hence compute active learning quality scores for).
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
        This argument is optional if you only want to get active learning scores for labeled examples (pass in ``pred_probs`` instead).

    Returns
    -------
    active_learning_scores : np.ndarray
        Similar to output as :py:func:`get_label_quality_scores <cleanlab.multiannotator.get_label_quality_scores>`.
    active_learning_scores_unlabeled : np.ndarray
        Similar to output as :py:func:`get_label_quality_scores <cleanlab.multiannotator.get_label_quality_scores>`.

    See Also
    --------
    get_active_learning_scores
    """

    assert_valid_pred_probs(
        pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled, ensemble=True
    )

    # compute multiannotator stats if labeled data is provided
    if pred_probs is not None:
        if labels_multiannotator is None:
            raise ValueError(
                "labels_multiannotator cannot be None when passing in pred_probs. ",
                "You can either provide labels_multiannotator to obtain active learning scores for the labeled examples, "
                "or just pass in pred_probs_unlabeled to get active learning scores for unlabeled examples.",
            )

        if isinstance(labels_multiannotator, pd.DataFrame):
            labels_multiannotator = (
                labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
            )
        elif not isinstance(labels_multiannotator, np.ndarray):
            raise ValueError(
                "labels_multiannotator must be either a NumPy array or Pandas DataFrame."
            )

        # check that labels_multiannotator is a 2D array
        if labels_multiannotator.ndim != 2:
            raise ValueError(
                "labels_multiannotator must be a 2D array or dataframe, "
                "each row represents an example and each column represents an annotator."
            )

        num_classes = get_num_classes(pred_probs=pred_probs[0])

        # if all examples are only labeled by a single annotator
        if (np.sum(~np.isnan(labels_multiannotator), axis=1) == 1).all():
            # do not temp scale for single annotator case, temperature is defined here for later use
            optimal_temp = np.full(len(pred_probs), 1.0)

            assert_valid_inputs_multiannotator(
                labels_multiannotator, pred_probs, ensemble=True, allow_single_label=True
            )

            avg_pred_probs = np.mean(pred_probs, axis=0)
            consensus_label = get_majority_vote_label(
                labels_multiannotator=labels_multiannotator,
                pred_probs=avg_pred_probs,
                verbose=False,
            )
            quality_of_consensus_labeled = get_label_quality_scores(consensus_label, avg_pred_probs)
            model_weight = np.full(len(pred_probs), 1)
            annotator_weight = np.full(labels_multiannotator.shape[1], 1)
            avg_annotator_weight = np.mean(annotator_weight)

        # examples are annotated by multiple annotators
        else:
            optimal_temp = np.full(len(pred_probs), np.NaN)
            for i, curr_pred_probs in enumerate(pred_probs):
                curr_optimal_temp = find_best_temp_scaler(labels_multiannotator, curr_pred_probs)
                pred_probs[i] = temp_scale_pred_probs(curr_pred_probs, curr_optimal_temp)
                optimal_temp[i] = curr_optimal_temp

            multiannotator_info = get_label_quality_multiannotator_ensemble(
                labels_multiannotator,
                pred_probs,
                return_annotator_stats=False,
                return_detailed_quality=False,
                return_weights=True,
            )

            quality_of_consensus_labeled = multiannotator_info["label_quality"][
                "consensus_quality_score"
            ]
            model_weight = multiannotator_info["model_weight"]
            annotator_weight = multiannotator_info["annotator_weight"]
            avg_annotator_weight = np.mean(annotator_weight)

        # compute scores for labeled data
        active_learning_scores = np.full(len(labels_multiannotator), np.nan)
        for i, annotator_labels in enumerate(labels_multiannotator):
            active_learning_scores[i] = np.average(
                (quality_of_consensus_labeled[i], 1 / num_classes),
                weights=(
                    np.sum(annotator_weight[~np.isnan(annotator_labels)]) + np.sum(model_weight),
                    avg_annotator_weight,
                ),
            )

    # no labeled data provided so do not estimate temperature and model/annotator weights
    elif pred_probs_unlabeled is not None:
        num_classes = get_num_classes(pred_probs=pred_probs_unlabeled[0])
        optimal_temp = np.full(len(pred_probs_unlabeled), 1.0)
        model_weight = np.full(len(pred_probs_unlabeled), 1)
        avg_annotator_weight = 1
        active_learning_scores = np.array([])

    else:
        raise ValueError(
            "pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two."
        )

    # compute scores for unlabeled data
    if pred_probs_unlabeled is not None:
        for i in range(len(pred_probs_unlabeled)):
            pred_probs_unlabeled[i] = temp_scale_pred_probs(
                pred_probs_unlabeled[i], optimal_temp[i]
            )

        avg_pred_probs_unlabeled = np.mean(pred_probs_unlabeled, axis=0)
        consensus_label_unlabeled = get_majority_vote_label(
            np.argmax(pred_probs_unlabeled, axis=2).T,
            avg_pred_probs_unlabeled,
        )
        modified_pred_probs_unlabeled = np.average(
            np.concatenate(
                (
                    pred_probs_unlabeled,
                    np.full(pred_probs_unlabeled.shape[1:], 1 / num_classes)[np.newaxis, :, :],
                )
            ),
            weights=np.concatenate((model_weight, np.array([avg_annotator_weight]))),
            axis=0,
        )

        active_learning_scores_unlabeled = get_label_quality_scores(
            consensus_label_unlabeled, modified_pred_probs_unlabeled
        )
    else:
        active_learning_scores_unlabeled = np.array([])

    return active_learning_scores, active_learning_scores_unlabeled

def get_majority_vote_label(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Returns the majority vote label for each example, aggregated from the labels given by multiple annotators.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    verbose : bool, optional
        Important warnings and other printed statements may be suppressed if ``verbose`` is set to ``False``.
    Returns
    -------
    consensus_label: np.ndarray
        An array of shape ``(N,)`` with the majority vote label aggregated from all annotators.

        In the event of majority vote ties, ties are broken in the following order:
        using the model ``pred_probs`` (if provided) and selecting the class with highest predicted probability,
        using the empirical class frequencies and selecting the class with highest frequency,
        using an initial annotator quality score and selecting the class that has been labeled by annotators with higher quality,
        and lastly by random selection.
    """

    if isinstance(labels_multiannotator, pd.DataFrame):
        annotator_ids = labels_multiannotator.columns
        labels_multiannotator = (
            labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
        )
    elif isinstance(labels_multiannotator, np.ndarray):
        annotator_ids = None
    else:
        raise ValueError("labels_multiannotator must be either a NumPy array or Pandas DataFrame.")

    if verbose:
        assert_valid_inputs_multiannotator(
            labels_multiannotator, pred_probs, annotator_ids=annotator_ids
        )

    if pred_probs is not None:
        num_classes = pred_probs.shape[1]
    else:
        num_classes = int(np.nanmax(labels_multiannotator) + 1)

    array_idx = np.arange(labels_multiannotator.shape[0])
    label_count = np.zeros((labels_multiannotator.shape[0], num_classes))
    for i in range(labels_multiannotator.shape[1]):
        not_nan_mask = ~np.isnan(labels_multiannotator[:, i])
        # Get the indexes where the label is not missing for the annotator i as int.
        label_index = labels_multiannotator[not_nan_mask, i].astype(int)
        # Increase the counts of those labels by 1.
        label_count[array_idx[not_nan_mask], label_index] += 1

    mode_labels_multiannotator = np.full(label_count.shape, np.nan)
    modes_mask = label_count == np.max(label_count, axis=1).reshape(-1, 1)
    insert_index = np.zeros(modes_mask.shape[0], dtype=int)
    for i in range(modes_mask.shape[1]):
        mode_index = np.where(modes_mask[:, i])[0]
        mode_labels_multiannotator[mode_index, insert_index[mode_index]] = i
        insert_index[mode_index] += 1

    majority_vote_label = np.full(len(labels_multiannotator), np.nan)
    label_mode_count = (~np.isnan(mode_labels_multiannotator)).sum(axis=1)

    # obtaining consensus using annotator majority vote
    mode_count_one_mask = label_mode_count == 1
    majority_vote_label[mode_count_one_mask] = mode_labels_multiannotator[mode_count_one_mask, 0]
    nontied_idx = array_idx[mode_count_one_mask]
    tied_idx = {
        i: label_mode[:count].astype(int)
        for i, label_mode, count in zip(
            array_idx[~mode_count_one_mask],
            mode_labels_multiannotator[~mode_count_one_mask, :],
            label_mode_count[~mode_count_one_mask],
        )
    }

    # tiebreak 1: using pred_probs (if provided)
    if pred_probs is not None and len(tied_idx) > 0:
        for idx, label_mode in tied_idx.copy().items():
            max_pred_probs = np.where(
                pred_probs[idx, label_mode] == np.max(pred_probs[idx, label_mode])
            )[0]
            if len(max_pred_probs) == 1:
                majority_vote_label[idx] = label_mode[max_pred_probs[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_pred_probs]

    # tiebreak 2: using empirical class frequencies
    # current tiebreak will select the minority class (to prevent larger class imbalance)
    if len(tied_idx) > 0:
        class_frequencies = label_count.sum(axis=0)
        for idx, label_mode in tied_idx.copy().items():
            min_frequency = np.where(
                class_frequencies[label_mode] == np.min(class_frequencies[label_mode])
            )[0]
            if len(min_frequency) == 1:
                majority_vote_label[idx] = label_mode[min_frequency[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[min_frequency]

    # tiebreak 3: using initial annotator quality scores
    if len(tied_idx) > 0:
        nontied_majority_vote_label = majority_vote_label[nontied_idx]
        nontied_labels_multiannotator = labels_multiannotator[nontied_idx]
        annotator_agreement_with_consensus = np.zeros(nontied_labels_multiannotator.shape[1])
        for i in range(len(annotator_agreement_with_consensus)):
            labels = nontied_labels_multiannotator[:, i]
            labels_mask = ~np.isnan(labels)
            if np.sum(labels_mask) == 0:
                annotator_agreement_with_consensus[i] = np.NaN
            else:
                annotator_agreement_with_consensus[i] = np.mean(
                    labels[labels_mask] == nontied_majority_vote_label[labels_mask]
                )

        # impute average annotator accuracy for any annotator that do not overlap with consensus
        nan_mask = np.isnan(annotator_agreement_with_consensus)
        avg_annotator_agreement = np.mean(annotator_agreement_with_consensus[~nan_mask])
        annotator_agreement_with_consensus[nan_mask] = avg_annotator_agreement

        for idx, label_mode in tied_idx.copy().items():
            label_quality_score = np.array(
                [
                    np.mean(
                        annotator_agreement_with_consensus[
                            np.where(labels_multiannotator[idx] == label)[0]
                        ]
                    )
                    for label in label_mode
                ]
            )
            max_score = np.where(label_quality_score == label_quality_score.max())[0]
            if len(max_score) == 1:
                majority_vote_label[idx] = label_mode[max_score[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_score]

    # if still tied, break by random selection
    if len(tied_idx) > 0:
        warnings.warn(
            f"breaking ties of examples {list(tied_idx.keys())} by random selection, you may want to set seed for reproducability"
        )
        for idx, label_mode in tied_idx.items():
            majority_vote_label[idx] = np.random.choice(label_mode)

    if verbose:
        # check if any classes no longer appear in the set of consensus labels
        check_consensus_label_classes(
            labels_multiannotator=labels_multiannotator,
            consensus_label=majority_vote_label,
            consensus_method="majority_vote",
        )

    return majority_vote_label.astype(int)

