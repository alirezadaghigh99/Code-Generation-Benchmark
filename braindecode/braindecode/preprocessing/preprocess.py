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

def exponential_moving_standardize(
    data: NDArray,
    factor_new: float = 0.001,
    init_block_size: int | None = None,
    eps: float = 1e-4,
):
    r"""Perform exponential moving standardization.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential moving variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{->v_t}, eps)`.


    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: np.ndarray (n_channels, n_times)
        Standardized data.
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(data[0:init_block_size], axis=i_time_axis, keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=i_time_axis, keepdims=True)
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(
            eps, init_std
        )
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T

def exponential_moving_demean(
    data: NDArray, factor_new: float = 0.001, init_block_size: int | None = None
):
    r"""Perform exponential moving demeanining.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Deman the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t)`.

    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Demean data before to this index with regular demeaning.

    Returns
    -------
    demeaned: np.ndarray (n_channels, n_times)
        Demeaned data.
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(data[0:init_block_size], axis=i_time_axis, keepdims=True)
        demeaned[0:init_block_size] = data[0:init_block_size] - init_mean
    return demeaned.T

class Preprocessor(object):
    """Preprocessor for an MNE Raw or Epochs object.

    Applies the provided preprocessing function to the data of a Raw or Epochs
    object.
    If the function is provided as a string, the method with that name will be
    used (e.g., 'pick_channels', 'filter', etc.).
    If it is provided as a callable and `apply_on_array` is True, the
    `apply_function` method of Raw and Epochs object will be used to apply the
    function on the internal arrays of Raw and Epochs.
    If `apply_on_array` is False, the callable must directly modify the Raw or
    Epochs object (e.g., by calling its method(s) or modifying its attributes).

    Parameters
    ----------
    fn: str or callable
        If str, the Raw/Epochs object must have a method with that name.
        If callable, directly apply the callable to the object.
    apply_on_array : bool
        Ignored if `fn` is not a callable. If True, the `apply_function` of Raw
        and Epochs object will be used to run `fn` on the underlying arrays
        directly. If False, `fn` must directly modify the Raw or Epochs object.
    kwargs:
        Keyword arguments to be forwarded to the MNE function.
    """

    def __init__(self, fn: Callable | str, *, apply_on_array: bool = True, **kwargs):
        if hasattr(fn, "__name__") and fn.__name__ == "<lambda>":
            warn("Preprocessing choices with lambda functions cannot be saved.")
        if callable(fn) and apply_on_array:
            channel_wise = kwargs.pop("channel_wise", False)
            picks = kwargs.pop("picks", None)
            n_jobs = kwargs.pop("n_jobs", 1)
            kwargs = dict(
                fun=partial(fn, **kwargs),
                channel_wise=channel_wise,
                picks=picks,
                n_jobs=n_jobs,
            )
            fn = "apply_function"
        self.fn = fn
        self.kwargs = kwargs

    def apply(self, raw_or_epochs: BaseRaw | BaseEpochs):
        try:
            self._try_apply(raw_or_epochs)
        except RuntimeError:
            # Maybe the function needs the data to be loaded and the data was
            # not loaded yet. Not all MNE functions need data to be loaded,
            # most importantly the 'crop' function can be lazily applied
            # without preloading data which can make the overall preprocessing
            # pipeline substantially faster.
            raw_or_epochs.load_data()
            self._try_apply(raw_or_epochs)

    def _try_apply(self, raw_or_epochs):
        if callable(self.fn):
            self.fn(raw_or_epochs, **self.kwargs)
        else:
            if not hasattr(raw_or_epochs, self.fn):
                raise AttributeError(f"MNE object does not have a {self.fn} method.")
            getattr(raw_or_epochs, self.fn)(**self.kwargs)

