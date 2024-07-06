def _make_dataset(
    directory,
    extensions=None,
    is_valid_file=None,
    pts_unit="sec",
    tqdm_args=None,
    num_workers: int = 0,
):
    """Returns a list of all video files, timestamps, and offsets.

    Args:
        directory:
            Root directory path (should not contain subdirectories).
        extensions:
            Tuple of valid extensions.
        is_valid_file:
            Used to find valid files.
        pts_unit:
            Unit of the timestamps.
        tqdm_args:
            arguments to pass to tqdm
        num_workers:
            number of workers to use for multithreading

    Returns:
        A list of video files, timestamps, frame offsets, and fps.

    """

    if tqdm_args is None:
        tqdm_args = {}
    if extensions is None:
        if is_valid_file is None:
            ValueError("Both extensions and is_valid_file cannot be None")
        else:
            _is_valid_file = is_valid_file
    else:

        def is_valid_file_extension(filepath):
            return filepath.lower().endswith(extensions)

        if is_valid_file is None:
            _is_valid_file = is_valid_file_extension
        else:

            def _is_valid_file(filepath):
                return is_valid_file_extension(filepath) and is_valid_file(filepath)

    # find all video instances (no subdirectories)
    video_instances = []

    def on_error(error):
        raise error

    for root, _, files in os.walk(directory, onerror=on_error):
        for fname in files:
            # skip invalid files
            if not _is_valid_file(os.path.join(root, fname)):
                continue

            # keep track of valid files
            path = os.path.join(root, fname)
            video_instances.append(path)

    # define loader to get the timestamps
    num_workers = min(num_workers, len(video_instances))
    if len(video_instances) == 1:
        num_workers = 0
    loader = DataLoader(
        _TimestampFpsFromVideosDataset(video_instances, pts_unit=pts_unit),
        num_workers=num_workers,
        batch_size=None,
        shuffle=False,
    )

    # actually load the data
    tqdm_args = dict(tqdm_args)
    tqdm_args.setdefault("unit", " video")
    tqdm_args.setdefault("desc", "Counting frames in videos")
    timestamps_fpss = list(tqdm(loader, **tqdm_args))
    timestamps, fpss = zip(*timestamps_fpss)

    # get frame offsets
    frame_counts = [len(ts) for ts in timestamps]
    offsets = [0] + list(np.cumsum(frame_counts[:-1]))

    return video_instances, timestamps, offsets, fpss

