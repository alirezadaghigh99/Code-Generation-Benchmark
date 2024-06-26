def get_images_in_labeling_jobs_of_specific_batch(
    all_labeling_jobs: List[dict],
    batch_id: str,
) -> int:
    """Get the number of images in labeling jobs of a specific batch.

    Args:
        all_labeling_jobs: All labeling jobs.
        batch_id: ID of the batch.

    Returns:
        The number of images in labeling jobs of the batch.

    """

    matching_jobs = []
    for labeling_job in all_labeling_jobs:
        if batch_id in labeling_job["sourceBatch"]:
            matching_jobs.append(labeling_job)
    return sum(job["numImages"] for job in matching_jobs)