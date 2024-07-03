class MOABBDataset(BaseConcatDataset):
    """A class for moabb datasets.

    Parameters
    ----------
    dataset_name: str
        name of dataset included in moabb to be fetched
    subject_ids: list(int) | int | None
        (list of) int of subject(s) to be fetched. If None, data of all
        subjects is fetched.
    dataset_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset when instantiating it.
    dataset_load_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset's load_data method.
        Allows using the moabb cache_config=None and
        process_pipeline=None.
    """

    def __init__(
        self,
        dataset_name: str,
        subject_ids: list[int] | int | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        dataset_load_kwargs: dict[str, Any] | None = None,
    ):
        # soft dependency on moabb
        from moabb import __version__ as moabb_version

        if moabb_version == "1.0.0":
            warnings.warn(
                "moabb version 1.0.0 generates incorrect annotations. "
                "Please update to another version, version 0.5 or 1.0.1 "
            )

        raws, description = fetch_data_with_moabb(
            dataset_name,
            subject_ids,
            dataset_kwargs,
            dataset_load_kwargs=dataset_load_kwargs,
        )
        all_base_ds = [
            BaseDataset(raw, row) for raw, (_, row) in zip(raws, description.iterrows())
        ]
        super().__init__(all_base_ds)def _fetch_and_unpack_moabb_data(dataset, subject_ids=None, dataset_load_kwargs=None):
    if dataset_load_kwargs is None:
        data = dataset.get_data(subject_ids)
    else:
        data = dataset.get_data(subjects=subject_ids, **dataset_load_kwargs)

    raws, subject_ids, session_ids, run_ids = [], [], [], []
    for subj_id, subj_data in data.items():
        for sess_id, sess_data in subj_data.items():
            for run_id, raw in sess_data.items():
                annots = _annotations_from_moabb_stim_channel(raw, dataset)
                raw.set_annotations(annots)
                raws.append(raw)
                subject_ids.append(subj_id)
                session_ids.append(sess_id)
                run_ids.append(run_id)
    description = pd.DataFrame(
        {"subject": subject_ids, "session": session_ids, "run": run_ids}
    )
    return raws, descriptiondef _find_dataset_in_moabb(dataset_name, dataset_kwargs=None):
    # soft dependency on moabb
    from moabb.datasets.utils import dataset_list

    for dataset in dataset_list:
        if dataset_name == dataset.__name__:
            # return an instance of the found dataset class
            if dataset_kwargs is None:
                return dataset()
            else:
                return dataset(**dataset_kwargs)
    raise ValueError(f"{dataset_name} not found in moabb datasets")