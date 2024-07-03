def preprocess(
    concat_ds: BaseConcatDataset,
    preprocessors: list[Preprocessor],
    save_dir: str | None = None,
    overwrite: bool = False,
    n_jobs: int | None = None,
    offset: int = 0,
):
    """Apply preprocessors to a concat dataset.

    Parameters
    ----------
    concat_ds: BaseConcatDataset
        A concat of BaseDataset or WindowsDataset datasets to be preprocessed.
    preprocessors: list(Preprocessor)
        List of Preprocessor objects to apply to the dataset.
    save_dir : str | None
        If a string, the preprocessed data will be saved under the specified
        directory and the datasets in ``concat_ds`` will be reloaded with
        `preload=False`.
    overwrite : bool
        When `save_dir` is provided, controls whether to delete the old
        subdirectories that will be written to under `save_dir`. If False and
        the corresponding subdirectories already exist, a ``FileExistsError``
        will be raised.
    n_jobs : int | None
        Number of jobs for parallel execution. See `joblib.Parallel` for
        a more detailed explanation.
    offset : int
        If provided, the integer is added to the id of the dataset in the
        concat. This is useful in the setting of very large datasets, where
        one dataset has to be processed and saved at a time to account for
        its original position.

    Returns
    -------
    BaseConcatDataset:
        Preprocessed dataset.
    """
    # In case of serialization, make sure directory is available before
    # preprocessing
    if save_dir is not None and not overwrite:
        _check_save_dir_empty(save_dir)

    if not isinstance(preprocessors, Iterable):
        raise ValueError("preprocessors must be a list of Preprocessor objects.")
    for elem in preprocessors:
        assert hasattr(elem, "apply"), "Preprocessor object needs an `apply` method."

    parallel_processing = (n_jobs is not None) and (n_jobs != 1)

    list_of_ds = Parallel(n_jobs=n_jobs)(
        delayed(_preprocess)(
            ds,
            i + offset,
            preprocessors,
            save_dir,
            overwrite,
            copy_data=(parallel_processing and (save_dir is None)),
        )
        for i, ds in enumerate(concat_ds.datasets)
    )

    if save_dir is not None:  # Reload datasets and replace in concat_ds
        ids_to_load = [i + offset for i in range(len(concat_ds.datasets))]
        concat_ds_reloaded = load_concat_dataset(
            save_dir,
            preload=False,
            target_name=None,
            ids_to_load=ids_to_load,
        )
        _replace_inplace(concat_ds, concat_ds_reloaded)
    else:
        if parallel_processing:  # joblib made copies
            _replace_inplace(concat_ds, BaseConcatDataset(list_of_ds))
        else:  # joblib did not make copies, the
            # preprocessing happened in-place
            # Recompute cumulative sizes as transforms might have changed them
            concat_ds.cumulative_sizes = concat_ds.cumsum(concat_ds.datasets)

    return concat_ds