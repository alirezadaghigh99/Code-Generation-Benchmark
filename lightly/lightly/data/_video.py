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

class VideoDataset(datasets.VisionDataset):
    """Implementation of a video dataset.

    The VideoDataset allows random reads from a video file without extracting
    all frames beforehand. This is more storage efficient but is slower.

    Attributes:
        root:
            Root directory path.
        extensions:
            Tuple of allowed extensions.
        transform:
            Function that takes a PIL image and returns transformed version
        target_transform:
            As transform but for targets
        is_valid_file:
            Used to check corrupt files
        exception_on_non_increasing_timestamp:
            If True, a NonIncreasingTimestampError is raised when trying to load
            a frame that has a timestamp lower or equal to the timestamps of
            previous frames in the same video.

    """

    def __init__(
        self,
        root,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        exception_on_non_increasing_timestamp=True,
        tqdm_args: Dict[str, Any] = None,
        num_workers: int = 0,
    ):
        super(VideoDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        videos, video_timestamps, offsets, fps = _make_dataset(
            self.root,
            extensions,
            is_valid_file,
            tqdm_args=tqdm_args,
            num_workers=num_workers,
        )

        if len(videos) == 0:
            msg = "Found 0 videos in folder: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.extensions = extensions
        self.backend = torchvision.get_video_backend()
        self.exception_on_non_increasing_timestamp = (
            exception_on_non_increasing_timestamp
        )

        self.videos = videos
        self.video_timestamps = video_timestamps
        self._length = sum((len(ts) for ts in self.video_timestamps))
        # Boolean value for every timestamp in self.video_timestamps. If True
        # the timestamp of the frame is non-increasing compared to timestamps of
        # previous frames in the video.
        self.video_timestamps_is_non_increasing = [
            _find_non_increasing_timestamps(timestamps)
            for timestamps in video_timestamps
        ]

        # offsets[i] indicates the index of the first frame of the i-th video.
        # e.g. for two videos of length 10 and 20, the offsets will be [0, 10].
        self.offsets = offsets
        self.fps = fps

        # Current VideoLoader instance and the corresponding video index. We
        # only keep track of the last accessed video as this is a good trade-off
        # between speed and memory requirements.
        # See https://github.com/lightly-ai/lightly/pull/702 for details.
        self._video_loader = None
        self._video_index = None

        # Keep unique reference of dataloader worker. We need this to avoid
        # accidentaly sharing VideoLoader instances between workers.
        self._worker_ref = None

        # Lock to prevent multiple threads creating a new VideoLoader at the
        # same time.
        self._video_loader_lock = threading.Lock()

    def __getitem__(self, index):
        """Returns item at index.

        Finds the video of the frame at index with the help of the frame
        offsets. Then, loads the frame from the video, applies the transforms,
        and returns the frame along with the index of the video (as target).

        For example, if there are two videos with 10 and 20 frames respectively
        in the input directory:

        Requesting the 5th sample returns the 5th frame from the first video and
        the target indicates the index of the source video which is 0.
        >>> dataset[5]
        >>> > <PIL Image>, 0

        Requesting the 20th sample returns the 10th frame from the second video
        and the target indicates the index of the source video which is 1.
        >>> dataset[20]
        >>> > <PIL Image>, 1

        Args:
            index:
                Index of the sample to retrieve.

        Returns:
            A tuple (sample, target) where target indicates the video index.

        Raises:
            IndexError:
                If index is out of bounds.
            VideoError:
                If the frame at the given index could not be loaded.

        """
        if index < 0 or index >= self.__len__():
            raise IndexError(
                f"Index {index} is out of bounds for VideoDataset"
                f" of size {self.__len__()}."
            )

        # each sample belongs to a video, to load the sample at index, we need
        # to find the video to which the sample belongs and then read the frame
        # from this video on the disk.
        i = len(self.offsets) - 1
        while self.offsets[i] > index:
            i = i - 1

        timestamp_idx = index - self.offsets[i]

        if (
            self.exception_on_non_increasing_timestamp
            and self.video_timestamps_is_non_increasing[i][timestamp_idx]
        ):
            raise NonIncreasingTimestampError(
                f"Frame {timestamp_idx} of video {self.videos[i]} has "
                f"a timestamp that is equal or lower than timestamps of previous "
                f"frames in the video. Trying to load this frame might result "
                f"in the wrong frame being returned. Set the VideoDataset.exception_on_non_increasing_timestamp"
                f"attribute to False to allow unsafe frame loading."
            )

        # find and return the frame as PIL image
        frame_timestamp = self.video_timestamps[i][timestamp_idx]
        video_loader = self._get_video_loader(i)
        sample = video_loader.read_frame(frame_timestamp)

        target = i
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """Returns the number of samples (frames) in the dataset.

        This can be precomputed, because self.video_timestamps is only
        set in the __init__
        """
        return self._length

    def get_filename(self, index):
        """Returns a filename for the frame at index.

        The filename is created from the video filename, the frame number, and
        the video format. The frame number will be zero padded to make sure
        all filenames have the same length and can easily be sorted.
        E.g. when retrieving a sample from the video
        `my_video.mp4` at frame 153, the filename will be:

        >>> my_video-153-mp4.png

        Args:
            index:
                Index of the frame to retrieve.

        Returns:
            The filename of the frame as described above.

        """
        if index < 0 or index >= self.__len__():
            raise IndexError(
                f"Index {index} is out of bounds for VideoDataset"
                f" of size {self.__len__()}."
            )

        # each sample belongs to a video, to load the sample at index, we need
        # to find the video to which the sample belongs and then read the frame
        # from this video on the disk.
        i = len(self.offsets) - 1
        while self.offsets[i] > index:
            i = i - 1

        # get filename of the video file
        video = self.videos[i]
        video_name, video_format = self._video_name_format(video)

        # get frame number
        frame_number = index - self.offsets[i]

        n_frames = self._video_frame_count(i)
        zero_padding = len(str(n_frames))

        return self._format_filename(
            video_name=video_name,
            video_format=video_format,
            frame_number=frame_number,
            zero_padding=zero_padding,
        )

    def get_filenames(self) -> List[str]:
        """Returns a list filenames for all frames in the dataset."""
        filenames = []
        for i, video in enumerate(self.videos):
            video_name, video_format = self._video_name_format(video)
            n_frames = self._video_frame_count(i)

            zero_padding = len(str(n_frames))
            for frame_number in range(n_frames):
                filenames.append(
                    self._format_filename(
                        video_name=video_name,
                        frame_number=frame_number,
                        video_format=video_format,
                        zero_padding=zero_padding,
                    )
                )
        return filenames

    def _video_frame_count(self, video_index: int) -> int:
        """Returns the number of frames in the video with the given index."""
        if video_index < len(self.offsets) - 1:
            n_frames = self.offsets[video_index + 1] - self.offsets[video_index]
        else:
            n_frames = len(self) - self.offsets[video_index]
        return n_frames

    def _video_name_format(self, video_filename: str) -> Tuple[str, str]:
        """Extracts name and format from the filename of the video.

        Returns:
            A (video_name, video_format) tuple where video_name is the filename
            relative to self.root and video_format is the file extension, for
            example 'mp4'.

        """
        video_filename = os.path.relpath(video_filename, self.root)
        splits = video_filename.split(".")
        video_format = splits[-1]
        video_name = ".".join(splits[:-1])
        return video_name, video_format

    def _format_filename(
        self,
        video_name: str,
        frame_number: int,
        video_format: str,
        zero_padding: int = 8,
        extension: str = "png",
    ) -> str:
        return f"{video_name}-{frame_number:0{zero_padding}}-{video_format}.{extension}"

    def _get_video_loader(self, video_index: int) -> VideoLoader:
        """Returns a video loader unique to the current dataloader worker."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Use a weakref instead of worker_info.id as the worker id is reused
            # by different workers across epochs.
            worker_ref = weakref.ref(worker_info)
            if worker_ref != self._worker_ref:
                # This worker has never accessed the dataset before, we have to
                # reset the video loader.
                self._video_loader = None
                self._video_index = None
                self._worker_ref = worker_ref

        with self._video_loader_lock:
            if video_index != self._video_index:
                video = self.videos[video_index]
                timestamps = self.video_timestamps[video_index]
                self._video_loader = VideoLoader(
                    video, timestamps, backend=self.backend
                )
                self._video_index = video_index

            return self._video_loader

