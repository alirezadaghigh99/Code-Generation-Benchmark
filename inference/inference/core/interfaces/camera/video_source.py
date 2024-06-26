def get_fps_if_tick_happens_now(fps_monitor: sv.FPSMonitor) -> float:
    if len(fps_monitor.all_timestamps) == 0:
        return 0.0
    min_reader_timestamp = fps_monitor.all_timestamps[0]
    now = time.monotonic()
    reader_taken_time = now - min_reader_timestamp
    return (len(fps_monitor.all_timestamps) + 1) / reader_taken_time