def _fetch_and_unpack_moabb_data(dataset, subject_ids=None, dataset_load_kwargs=None):
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
    return raws, description

def _find_dataset_in_moabb(dataset_name, dataset_kwargs=None):
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

