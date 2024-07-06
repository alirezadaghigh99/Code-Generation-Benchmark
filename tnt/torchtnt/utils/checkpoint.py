def from_str(cls, checkpoint_path: str) -> "CheckpointPath":
        """
        Given a directory path, try to parse it and extract the checkpoint data.
        The expected format is: <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>,
        where the metric name and value are optional.

        Args:
            checkpoint_path: The path to the checkpoint directory.

        Returns:
            A CheckpointPath instance if the path is valid, otherwise None.

        Raises:
            ValueError: If the path is malformed and can't be parsed.
        """
        ckpt_path = cls.__new__(cls)
        ckpt_path._populate_from_str(checkpoint_path)
        return ckpt_path

def _retrieve_checkpoint_dirpaths(
    dirpath: str,
    metadata_fname: Optional[str],
    metric_name: Optional[str] = None,
) -> List[CheckpointPath]:
    """
    Given a parent directory where checkpoints are saved, return the unsorted checkpoint subdirectories

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        metric_name: Name of the metric that must exist in checkpoint name.
    """

    fs, _ = url_to_fs(dirpath)

    if not fs.exists(dirpath):
        logger.warning(f"Input dirpath doesn't exist: {dirpath}")
        return []

    contents = fs.ls(dirpath, detail=True)
    contents = [item["name"] for item in contents if item["type"] == "directory"]
    if len(contents) == 0:
        logger.warning(f"Input dirpath doesn't contain any subdirectories: {dirpath}")
        return []

    # Parse the valid checkpoint directories
    candidate_checkpoints: List[CheckpointPath] = []
    for candidate_dirpath in contents:
        try:
            ckpt = CheckpointPath.from_str(candidate_dirpath)
        except ValueError:
            continue

        # If a metric was provided, keep only the checkpoints tracking it
        if metric_name and not (
            ckpt.metric_data and ckpt.metric_data.name == metric_name
        ):
            continue

        candidate_checkpoints.append(ckpt)

    if not metadata_fname:
        # return early as we don't need to filter out any paths
        return candidate_checkpoints

    # Iterate through all files and directories in the specified directory
    # and check if metedata is present or not
    valid_ckpt_dirpaths: List[CheckpointPath] = []
    for candidate in candidate_checkpoints:
        if not _metadata_exists(fs, candidate.path, metadata_fname):
            logger.warning(
                f"Snapshot metadata is missing from {candidate}! Skipping this path"
            )
            continue

        valid_ckpt_dirpaths.append(candidate)

    return valid_ckpt_dirpaths

def get_checkpoint_dirpaths(
    dirpath: str,
    metadata_fname: Optional[str] = None,
    metric_name: Optional[str] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> List[CheckpointPath]:
    """
    Given a parent directory where checkpoints are saved, returns the checkpoint subdirectories.
    The order of the checkpoints is not guaranteed.

    The checkpoint paths are assumed to have the following format: <dirpath>/epoch_<epoch>_step_<step>
    If a metric_name is provided the format should be <dirpath>/epoch_<epoch>_step_<step>_<metric_name>=<metric_value>
    This will always be the case if the CheckpointManager class is used to produce their names.

    Args:
        dirpath: parent directory where checkpoints are saved.
        metadata_fname: Checks if metadata file is present in checkpoint, disregards if it does not exist.
        metric_name: fetches all the checkpoint directories containing the metric name only.
        process_group: the process group on which the ranks will communicate on. default: ``None`` (the entire world)

    Note:
        When doing distributed training, only rank 0 will read the file system. The result will be broadcasted to all ranks.
        gloo process groups are recommended over nccl.
    """

    return _retrieve_checkpoint_dirpaths(dirpath, metadata_fname, metric_name)

