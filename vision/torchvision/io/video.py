def read_video_timestamps(filename: str, pts_unit: str = "pts") -> Tuple[List[int], Optional[float]]:
    """
    List the video frames timestamps.

    Note that the function decodes the whole video frame-by-frame.

    Args:
        filename (str): path to the video file
        pts_unit (str, optional): unit in which timestamp values will be returned
            either 'pts' or 'sec'. Defaults to 'pts'.

    Returns:
        pts (List[int] if pts_unit = 'pts', List[Fraction] if pts_unit = 'sec'):
            presentation timestamps for each one of the frames in the video.
        video_fps (float, optional): the frame rate for the video

    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_video_timestamps)
    from torchvision import get_video_backend

    if get_video_backend() != "pyav":
        return _video_opt._read_video_timestamps(filename, pts_unit)

    _check_av_available()

    video_fps = None
    pts = []

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            if container.streams.video:
                video_stream = container.streams.video[0]
                video_time_base = video_stream.time_base
                try:
                    pts = _decode_video_timestamps(container)
                except av.AVError:
                    warnings.warn(f"Failed decoding frames for file {filename}")
                video_fps = float(video_stream.average_rate)
    except av.AVError as e:
        msg = f"Failed to open container for {filename}; Caught error: {e}"
        warnings.warn(msg, RuntimeWarning)

    pts.sort()

    if pts_unit == "sec":
        pts = [x * video_time_base for x in pts]

    return pts, video_fps

def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file. If using the pyav backend, this can be whatever ``av.open`` accepts.
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_video)

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    from torchvision import get_video_backend

    if get_video_backend() != "pyav":
        if not os.path.exists(filename):
            raise RuntimeError(f"File not found: {filename}")
        vframes, aframes, info = _video_opt._read_video(filename, start_pts, end_pts, pts_unit)
    else:
        _check_av_available()

        if end_pts is None:
            end_pts = float("inf")

        if end_pts < start_pts:
            raise ValueError(
                f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
            )

        info = {}
        video_frames = []
        audio_frames = []
        audio_timebase = _video_opt.default_timebase

        try:
            with av.open(filename, metadata_errors="ignore") as container:
                if container.streams.audio:
                    audio_timebase = container.streams.audio[0].time_base
                if container.streams.video:
                    video_frames = _read_from_stream(
                        container,
                        start_pts,
                        end_pts,
                        pts_unit,
                        container.streams.video[0],
                        {"video": 0},
                    )
                    video_fps = container.streams.video[0].average_rate
                    # guard against potentially corrupted files
                    if video_fps is not None:
                        info["video_fps"] = float(video_fps)

                if container.streams.audio:
                    audio_frames = _read_from_stream(
                        container,
                        start_pts,
                        end_pts,
                        pts_unit,
                        container.streams.audio[0],
                        {"audio": 0},
                    )
                    info["audio_fps"] = container.streams.audio[0].rate

        except av.AVError:
            # TODO raise a warning?
            pass

        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
        aframes_list = [frame.to_ndarray() for frame in audio_frames]

        if vframes_list:
            vframes = torch.as_tensor(np.stack(vframes_list))
        else:
            vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

        if aframes_list:
            aframes = np.concatenate(aframes_list, 1)
            aframes = torch.as_tensor(aframes)
            if pts_unit == "sec":
                start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
                if end_pts != float("inf"):
                    end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
            aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
        else:
            aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info

def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(write_video)
    _check_av_available()
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = np.round(fps)

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = container.streams.audio[0].format.name

            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(audio_array, format=audio_sample_fmt, layout=audio_layout)

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

