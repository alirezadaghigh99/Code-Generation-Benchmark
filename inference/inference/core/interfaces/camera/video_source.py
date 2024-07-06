def get_from_queue(
    queue: Queue,
    timeout: Optional[float] = None,
    on_successful_read: Callable[[], None] = lambda: None,
    purge: bool = False,
) -> Optional[Any]:
    """
    Function is supposed to take element from the queue waiting on the first element to appear using `timeout`
    parameter. One may ask to go to the very last element of the queue and return it - then `purge` should be set
    to True. No additional wait on new elements to appear happen and the purge stops once queue is free returning last
    element consumed.
    queue.task_done() and on_successful_read(...) will be called on each received element.
    """
    result = None
    if queue.empty() or not purge:
        try:
            result = queue.get(timeout=timeout)
            queue.task_done()
            on_successful_read()
        except Empty:
            pass
    while not queue.empty() and purge:
        result = queue.get()
        queue.task_done()
        on_successful_read()
    return result

def get_fps_if_tick_happens_now(fps_monitor: sv.FPSMonitor) -> float:
    if len(fps_monitor.all_timestamps) == 0:
        return 0.0
    min_reader_timestamp = fps_monitor.all_timestamps[0]
    now = time.monotonic()
    reader_taken_time = now - min_reader_timestamp
    return (len(fps_monitor.all_timestamps) + 1) / reader_taken_time

def decode_video_frame_to_buffer(
    frame_timestamp: datetime,
    frame_id: int,
    video: VideoFrameProducer,
    buffer: Queue,
    decoding_pace_monitor: sv.FPSMonitor,
    source_id: Optional[int],
) -> bool:
    success, image = video.retrieve()
    if not success:
        return False
    decoding_pace_monitor.tick()
    video_frame = VideoFrame(
        image=image,
        frame_id=frame_id,
        frame_timestamp=frame_timestamp,
        source_id=source_id,
    )
    buffer.put(video_frame)
    return True

