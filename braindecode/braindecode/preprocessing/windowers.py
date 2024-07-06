def create_windows_from_events(
    concat_ds: BaseConcatDataset,
    trial_start_offset_samples: int = 0,
    trial_stop_offset_samples: int = 0,
    window_size_samples: int | None = None,
    window_stride_samples: int | None = None,
    drop_last_window: bool = False,
    mapping: dict[str, int] | None = None,
    preload: bool = False,
    drop_bad_windows: bool | None = None,
    picks: str | ArrayLike | slice | None = None,
    reject: dict[str, float] | None = None,
    flat: dict[str, float] | None = None,
    on_missing: str = "error",
    accepted_bads_ratio: float = 0.0,
    use_mne_epochs: bool | None = None,
    n_jobs: int = 1,
    verbose: bool | str | int | None = "error",
):
    """Create windows based on events in mne.Raw.

    This function extracts windows of size window_size_samples in the interval
    [trial onset + trial_start_offset_samples, trial onset + trial duration +
    trial_stop_offset_samples] around each trial, with a separation of
    window_stride_samples between consecutive windows. If the last window
    around an event does not end at trial_stop_offset_samples and
    drop_last_window is set to False, an additional overlapping window that
    ends at trial_stop_offset_samples is created.

    Windows are extracted from the interval defined by the following::

                                                trial onset +
                        trial onset                duration
        |--------------------|------------------------|-----------------------|
        trial onset -                                             trial onset +
        trial_start_offset_samples                                   duration +
                                                    trial_stop_offset_samples

    Parameters
    ----------
    concat_ds: BaseConcatDataset
        A concat of base datasets each holding raw and description.
    trial_start_offset_samples: int
        Start offset from original trial onsets, in samples. Defaults to zero.
    trial_stop_offset_samples: int
        Stop offset from original trial stop, in samples. Defaults to zero.
    window_size_samples: int | None
        Window size. If None, the window size is inferred from the original
        trial size of the first trial and trial_start_offset_samples and
        trial_stop_offset_samples.
    window_stride_samples: int | None
        Stride between windows, in samples. If None, the window stride is
        inferred from the original trial size of the first trial and
        trial_start_offset_samples and trial_stop_offset_samples.
    drop_last_window: bool
        If False, an additional overlapping window that ends at
        trial_stop_offset_samples will be extracted around each event when the
        last window does not end exactly at trial_stop_offset_samples.
    mapping: dict(str: int)
        Mapping from event description to numerical target value.
    preload: bool
        If True, preload the data of the Epochs objects. This is useful to
        reduce disk reading overhead when returning windows in a training
        scenario, however very large data might not fit into memory.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well.
    picks: str | list | slice | None
        Channels to include. If None, all available channels are used. See
        mne.Epochs.
    reject: dict | None
        Epoch rejection parameters based on peak-to-peak amplitude. If None, no
        rejection is done based on peak-to-peak amplitude. See mne.Epochs.
    flat: dict | None
        Epoch rejection parameters based on flatness of signals. If None, no
        rejection based on flatness is done. See mne.Epochs.
    on_missing: str
        What to do if one or several event ids are not found in the recording.
        Valid keys are ‘error’ | ‘warning’ | ‘ignore’. See mne.Epochs.
    accepted_bads_ratio: float, optional
        Acceptable proportion of trials with inconsistent length in a raw. If
        the number of trials whose length is exceeded by the window size is
        smaller than this, then only the corresponding trials are dropped, but
        the computation continues. Otherwise, an error is raised. Defaults to
        0.0 (raise an error).
    use_mne_epochs: bool
        If False, return EEGWindowsDataset objects.
        If True, return mne.Epochs objects encapsulated in WindowsDataset objects,
        which is substantially slower that EEGWindowsDataset.
    n_jobs: int
        Number of jobs to use to parallelize the windowing.
    verbose: bool | str | int | None
        Control verbosity of the logging output when calling mne.Epochs.

    Returns
    -------
    windows_datasets: BaseConcatDataset
        Concatenated datasets of WindowsDataset containing the extracted windows.
    """
    _check_windowing_arguments(
        trial_start_offset_samples,
        trial_stop_offset_samples,
        window_size_samples,
        window_stride_samples,
    )

    # If user did not specify mapping, we extract all events from all datasets
    # and map them to increasing integers starting from 0
    infer_mapping = mapping is None
    mapping = dict() if infer_mapping else mapping
    infer_window_size_stride = window_size_samples is None

    if drop_bad_windows is not None:
        warnings.warn(
            "Drop bad windows only has an effect if mne epochs are created, "
            "and this argument may be removed in the future."
        )

    use_mne_epochs = _get_use_mne_epochs(
        use_mne_epochs, reject, picks, flat, drop_bad_windows
    )
    if use_mne_epochs and drop_bad_windows is None:
        drop_bad_windows = True

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_windows_from_events)(
            ds,
            infer_mapping,
            infer_window_size_stride,
            trial_start_offset_samples,
            trial_stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            drop_last_window,
            mapping,
            preload,
            drop_bad_windows,
            picks,
            reject,
            flat,
            on_missing,
            accepted_bads_ratio,
            verbose,
            use_mne_epochs,
        )
        for ds in concat_ds.datasets
    )
    return BaseConcatDataset(list_of_windows_ds)

def create_fixed_length_windows(
    concat_ds: BaseConcatDataset,
    start_offset_samples: int = 0,
    stop_offset_samples: int | None = None,
    window_size_samples: int | None = None,
    window_stride_samples: int | None = None,
    drop_last_window: bool | None = None,
    mapping: dict[str, int] | None = None,
    preload: bool = False,
    picks: str | ArrayLike | slice | None = None,
    reject: dict[str, float] | None = None,
    flat: dict[str, float] | None = None,
    targets_from: str = "metadata",
    last_target_only: bool = True,
    lazy_metadata: bool = False,
    on_missing: str = "error",
    n_jobs: int = 1,
    verbose: bool | str | int | None = "error",
):
    """Windower that creates sliding windows.

    Parameters
    ----------
    concat_ds: ConcatDataset
        A concat of base datasets each holding raw and description.
    start_offset_samples: int
        Start offset from beginning of recording in samples.
    stop_offset_samples: int | None
        Stop offset from beginning of recording in samples. If None, set to be
        the end of the recording.
    window_size_samples: int | None
        Window size in samples. If None, set to be the maximum possible window size, ie length of
        the recording, once offsets are accounted for.
    window_stride_samples: int | None
        Stride between windows in samples. If None, set to be equal to winddow_size_samples, so
        windows will not overlap.
    drop_last_window: bool | None
        Whether or not have a last overlapping window, when windows do not
        equally divide the continuous signal. Must be set to a bool if window size and stride are
        not None.
    mapping: dict(str: int)
        Mapping from event description to target value.
    preload: bool
        If True, preload the data of the Epochs objects.
    picks: str | list | slice | None
        Channels to include. If None, all available channels are used. See
        mne.Epochs.
    reject: dict | None
        Epoch rejection parameters based on peak-to-peak amplitude. If None, no
        rejection is done based on peak-to-peak amplitude. See mne.Epochs.
    flat: dict | None
        Epoch rejection parameters based on flatness of signals. If None, no
        rejection based on flatness is done. See mne.Epochs.
    lazy_metadata: bool
        If True, metadata is not computed immediately, but only when accessed
        by using the _LazyDataFrame (experimental).
    on_missing: str
        What to do if one or several event ids are not found in the recording.
        Valid keys are ‘error’ | ‘warning’ | ‘ignore’. See mne.Epochs.
    n_jobs: int
        Number of jobs to use to parallelize the windowing.
    verbose: bool | str | int | None
        Control verbosity of the logging output when calling mne.Epochs.

    Returns
    -------
    windows_datasets: BaseConcatDataset
        Concatenated datasets of WindowsDataset containing the extracted windows.
    """
    stop_offset_samples, drop_last_window = (
        _check_and_set_fixed_length_window_arguments(
            start_offset_samples,
            stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            drop_last_window,
            lazy_metadata,
        )
    )

    # check if recordings are of different lengths
    lengths = np.array([ds.raw.n_times for ds in concat_ds.datasets])
    if (np.diff(lengths) != 0).any() and window_size_samples is None:
        warnings.warn("Recordings have different lengths, they will not be batch-able!")
    if (window_size_samples is not None) and any(window_size_samples > lengths):
        raise ValueError(
            f"Window size {window_size_samples} exceeds trial "
            f"duration {lengths.min()}."
        )

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_fixed_length_windows)(
            ds,
            start_offset_samples,
            stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            drop_last_window,
            mapping,
            preload,
            picks,
            reject,
            flat,
            targets_from,
            last_target_only,
            lazy_metadata,
            on_missing,
            verbose,
        )
        for ds in concat_ds.datasets
    )
    return BaseConcatDataset(list_of_windows_ds)

