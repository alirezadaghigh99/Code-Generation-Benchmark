def calculate_feature_importance_or_none(
        model: t.Any,
        dataset: t.Union['tabular.Dataset', pd.DataFrame],
        model_classes,
        observed_classes,
        task_type,
        force_permutation: bool = False,
        permutation_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Tuple[t.Optional[pd.Series], t.Optional[str]]:
    """Calculate features effect on the label or None if the input is incorrect.

    Parameters
    ----------
    model : t.Any
        a fitted model
    dataset : t.Union['tabular.Dataset', pd.DataFrame]
        dataset used to fit the model
    model_classes
        possible classes output for model. None for regression tasks.
    observed_classes
        Observed classes in the data. None for regression tasks.
    task_type
        The task type of the model.
    force_permutation : bool , default: False
        force permutation importance calculation
    permutation_kwargs : t.Optional[t.Dict[str, t.Any]] , default: None
        kwargs for permutation importance calculation

    Returns
    -------
    feature_importance, calculation_type : t.Tuple[t.Optional[pd.Series], str]]
        features importance normalized to 0-1 indexed by feature names, or None if the input is incorrect
        Tuple of the features importance and the calculation type
        (types: `permutation_importance`, `feature_importances_`, `coef_`)
    """
    try:
        if model is None:
            return None
        # calculate feature importance if dataset has a label and the model is fitted on it
        fi, calculation_type = _calculate_feature_importance(
            model=model,
            dataset=dataset,
            model_classes=model_classes,
            observed_classes=observed_classes,
            task_type=task_type,
            force_permutation=force_permutation,
            permutation_kwargs=permutation_kwargs,
        )

        return fi, calculation_type
    except (
            errors.DeepchecksValueError,
            errors.NumberOfFeaturesLimitError,
            errors.DeepchecksTimeoutError,
            errors.ModelValidationError,
            errors.DatasetValidationError,
            errors.DeepchecksSkippedFeatureImportance
    ) as error:
        # DeepchecksValueError:
        #     if model validation failed;
        #     if it was not possible to calculate features importance;
        # NumberOfFeaturesLimitError:
        #     if the number of features limit were exceeded;
        # DatasetValidationError:
        #     if dataset did not meet requirements
        # ModelValidationError:
        #     if wrong type of model was provided;
        #     if function failed to predict on model;
        get_logger().warning('Features importance was not calculated:\n%s', error)
        return None, None

