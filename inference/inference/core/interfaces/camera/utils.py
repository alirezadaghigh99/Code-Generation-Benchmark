def get_video_frames_generator(
    video: Union[VideoSource, str, int],
    max_fps: Optional[Union[float, int]] = None,
    limiter_strategy: Optional[FPSLimiterStrategy] = None,
) -> Generator[VideoFrame, None, None]:
    """
    Util function to create a frames generator from `VideoSource` with possibility to
    limit FPS of consumed frames and dictate what to do if frames are produced to fast.

    Args:
        video (Union[VideoSource, str, int]): Either instance of VideoSource or video reference accepted
            by VideoSource.init(...)
        max_fps (Optional[Union[float, int]]): value of maximum FPS rate of generated frames - can be used to limit
            generation frequency
        limiter_strategy (Optional[FPSLimiterStrategy]): strategy used to deal with frames decoding exceeding
            limit of `max_fps`. By default - for files, in the interest of processing all frames -
            generation will be awaited, for streams - frames will be dropped on the floor.
    Returns: generator of `VideoFrame`

    Example:
        ```python
        from inference.core.interfaces.camera.utils import get_video_frames_generator

        for frame in get_video_frames_generator(
            video="./some.mp4",
            max_fps=50,
        ):
             pass
        ```
    """
    is_managed_source = False
    if issubclass(type(video), str) or issubclass(type(video), int):
        video = VideoSource.init(
            video_reference=video,
        )
        video.start()
        is_managed_source = True
    if max_fps is None:
        yield from video
        if is_managed_source:
            video.terminate(purge_frames_buffer=True)
        return None
    limiter_strategy = resolve_limiter_strategy(
        explicitly_defined_strategy=limiter_strategy,
        source_properties=video.describe_source().source_properties,
    )
    yield from limit_frame_rate(
        frames_generator=video, max_fps=max_fps, strategy=limiter_strategy
    )
    if is_managed_source:
        video.terminate(purge_frames_buffer=True)
    return None