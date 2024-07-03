def load_concat_dataset(path, preload, ids_to_load=None, target_name=None, n_jobs=1):
    """Load a stored BaseConcatDataset of BaseDatasets or WindowsDatasets from
    files.

    Parameters
    ----------
    path: str | pathlib.Path
        Path to the directory of the .fif / -epo.fif and .json files.
    preload: bool
        Whether to preload the data.
    ids_to_load: list of int | None
        Ids of specific files to load.
    target_name: str | list | None
        Load specific description column as target. If not given, take saved
        target name.
    n_jobs: int
        Number of jobs to be used to read files in parallel.

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    # Make sure we always work with a pathlib.Path
    path = Path(path)

    # if we encounter a dataset that was saved in 'the old way', call the
    # corresponding 'old' loading function
    if _is_outdated_saved(path):
        warnings.warn(
            "The way your dataset was saved is deprecated by now. "
            "Please save it again using dataset.save().",
            UserWarning,
        )
        return _outdated_load_concat_dataset(
            path=path, preload=preload, ids_to_load=ids_to_load, target_name=target_name
        )

    # else we have a dataset saved in the new way with subdirectories in path
    # for every dataset with description.json and -epo.fif or -raw.fif,
    # target_name.json, raw_preproc_kwargs.json, window_kwargs.json,
    # window_preproc_kwargs.json
    if ids_to_load is None:
        ids_to_load = [p.name for p in path.iterdir()]
        ids_to_load = sorted(ids_to_load, key=lambda i: int(i))
    ids_to_load = [str(i) for i in ids_to_load]
    first_raw_fif_path = path / ids_to_load[0] / f"{ids_to_load[0]}-raw.fif"
    is_raw = first_raw_fif_path.exists()
    metadata_path = path / ids_to_load[0] / "metadata_df.pkl"
    has_stored_windows = metadata_path.exists()

    # Parallelization of mne.read_epochs with preload=False fails with
    # 'TypeError: cannot pickle '_io.BufferedReader' object'.
    # So ignore n_jobs in that case and load with a single job.
    if not is_raw and n_jobs != 1:
        warnings.warn(
            "Parallelized reading with `preload=False` is not supported for "
            "windowed data. Will use `n_jobs=1`.",
            UserWarning,
        )
        n_jobs = 1
    datasets = Parallel(n_jobs)(
        delayed(_load_parallel)(path, i, preload, is_raw, has_stored_windows)
        for i in ids_to_load
    )
    return BaseConcatDataset(datasets)