def filter_gt_instances(batch_data_samples: SampleList,
                        score_thr: float = None,
                        wh_thr: tuple = None):
    """Filter ground truth (GT) instances by score and/or size.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        score_thr (float): The score filter threshold.
        wh_thr (tuple):  Minimum width and height of bbox.

    Returns:
        SampleList: The Data Samples filtered by score and/or size.
    """

    if score_thr is not None:
        batch_data_samples = _filter_gt_instances_by_score(
            batch_data_samples, score_thr)
    if wh_thr is not None:
        batch_data_samples = _filter_gt_instances_by_size(
            batch_data_samples, wh_thr)
    return batch_data_samples