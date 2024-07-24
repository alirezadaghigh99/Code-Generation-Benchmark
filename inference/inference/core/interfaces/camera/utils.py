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

def limit_frame_rate(
    frames_generator: Iterable[T],
    max_fps: Union[float, int],
    strategy: FPSLimiterStrategy,
) -> Generator[T, None, None]:
    rate_limiter = RateLimiter(desired_fps=max_fps)
    for frame_data in frames_generator:
        delay = rate_limiter.estimate_next_action_delay()
        if delay <= 0.0:
            rate_limiter.tick()
            yield frame_data
            continue
        if strategy is FPSLimiterStrategy.WAIT:
            time.sleep(delay)
            rate_limiter.tick()
            yield frame_data

def _prepare_video_sources(
    videos: List[Union[VideoSource, str, int]],
    force_stream_reconnection: bool,
) -> VideoSources:
    all_sources: List[VideoSource] = []
    managed_sources: List[VideoSource] = []
    minimal_free_source_id = _find_free_source_identifier(videos=videos)
    try:
        for video in videos:
            if issubclass(type(video), str) or issubclass(type(video), int):
                video = VideoSource.init(
                    video_reference=video, source_id=minimal_free_source_id
                )
                minimal_free_source_id += 1
                video.start()
                managed_sources.append(video)
            all_sources.append(video)
    except Exception as e:
        for video in managed_sources:
            try:
                video.terminate(
                    wait_on_frames_consumption=False, purge_frames_buffer=True
                )
            except Exception:
                # passing inner termination error
                pass
        raise e
    allow_reconnection = _establish_sources_reconnection_rules(
        all_sources=all_sources,
        force_stream_reconnection=force_stream_reconnection,
    )
    return VideoSources(
        all_sources=all_sources,
        allow_reconnection=allow_reconnection,
        managed_sources=managed_sources,
    )

def negotiate_rate_limiter_strategy_for_multiple_sources(
    video_sources: List[VideoSource],
) -> FPSLimiterStrategy:
    source_types_statuses = {
        s.describe_source().source_properties.is_file for s in video_sources
    }
    if len(source_types_statuses) == 2:
        logger.warning(
            f"`InferencePipeline` started with FPS limit rate. Detected both files and video streams as video sources. "
            f"Rate limiter cannot satisfy both - choosing `FPSLimiterStrategy.DROP` which may drop file sources frames "
            f"that would not happen if only video files are submitted into processing."
        )
        return FPSLimiterStrategy.DROP
    if True in source_types_statuses:
        return FPSLimiterStrategy.WAIT
    return FPSLimiterStrategy.DROP

def resolve_limiter_strategy(
    explicitly_defined_strategy: Optional[FPSLimiterStrategy],
    source_properties: Optional[SourceProperties],
) -> FPSLimiterStrategy:
    if explicitly_defined_strategy is not None:
        return explicitly_defined_strategy
    limiter_strategy = FPSLimiterStrategy.DROP
    if source_properties is not None and source_properties.is_file:
        limiter_strategy = FPSLimiterStrategy.WAIT
    return limiter_strategy

def _attempt_reconnect(
    video_source: VideoSource,
    should_stop: Callable[[], bool],
    on_reconnection_failure: Callable[[Optional[int], SourceConnectionError], None],
    on_reconnection_success: Callable[[], None],
    on_fatal_error: Callable[[], None],
) -> None:
    succeeded = False
    while not should_stop() and not succeeded:
        try:
            video_source.restart(wait_on_frames_consumption=False)
            succeeded = True
            on_reconnection_success()
        except SourceConnectionError as error:
            on_reconnection_failure(video_source.source_id, error)
            if should_stop():
                return None
            logger.warning(
                f"Could not connect to video source. Retrying in {RESTART_ATTEMPT_DELAY}s..."
            )
            time.sleep(RESTART_ATTEMPT_DELAY)
        except Exception as error:
            logger.warning(
                f"Fatal error in re-connection to source: {video_source.source_id}. Details: {error}"
            )
            on_fatal_error()
            break

def _establish_sources_reconnection_rules(
    all_sources: List[VideoSource], force_stream_reconnection: bool
) -> List[bool]:
    result = []
    for video_source in all_sources:
        source_properties = video_source.describe_source().source_properties
        if source_properties is None:
            result.append(False)
        else:
            result.append(not source_properties.is_file and force_stream_reconnection)
    return result

def _find_free_source_identifier(videos: List[Union[VideoSource, str, int]]) -> int:
    minimal_free_source_id = [
        v.source_id if v.source_id is not None else -1
        for v in videos
        if issubclass(type(v), VideoSource)
    ]
    if len(minimal_free_source_id) == 0:
        minimal_free_source_id = -1
    else:
        minimal_free_source_id = max(minimal_free_source_id)
    minimal_free_source_id += 1
    return minimal_free_source_id

class RateLimiter:
    """
    Implements rate upper-bound rate limiting by ensuring estimate_next_tick_delay()
    to be at min 1 / desired_fps, not letting the client obeying outcomes to exceed
    assumed rate.
    """

    def __init__(self, desired_fps: Union[float, int]):
        self._desired_fps = max(desired_fps, MINIMAL_FPS)
        self._last_tick: Optional[float] = None

    def tick(self) -> None:
        self._last_tick = time.monotonic()

    def estimate_next_action_delay(self) -> float:
        if self._last_tick is None:
            return 0.0
        desired_delay = 1 / self._desired_fps
        time_since_last_tick = time.monotonic() - self._last_tick
        return max(desired_delay - time_since_last_tick, 0.0)

