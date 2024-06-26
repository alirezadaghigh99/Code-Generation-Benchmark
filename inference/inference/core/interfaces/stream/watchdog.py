class BasePipelineWatchDog(PipelineWatchDog):
    """
    Implementation to be used from single inference thread, as it keeps
    state assumed to represent status of consecutive stage of prediction process
    in latency monitor.
    """

    def __init__(self):
        super().__init__()
        self._video_sources: Optional[List[VideoSource]] = None
        self._inference_throughput_monitor = sv.FPSMonitor()
        self._latency_monitors: Dict[Optional[int], LatencyMonitor] = {}
        self._stream_updates = deque(maxlen=MAX_UPDATES_CONTEXT)

    def register_video_sources(self, video_sources: List[VideoSource]) -> None:
        self._video_sources = video_sources
        for source in video_sources:
            self._latency_monitors[source.source_id] = LatencyMonitor(
                source_id=source.source_id
            )

    def on_status_update(self, status_update: StatusUpdate) -> None:
        if status_update.severity.value <= UpdateSeverity.DEBUG.value:
            return None
        self._stream_updates.append(status_update)

    def on_model_inference_started(self, frames: List[VideoFrame]) -> None:
        for frame in frames:
            self._latency_monitors[frame.source_id].register_inference_start(
                frame_timestamp=frame.frame_timestamp,
                frame_id=frame.frame_id,
            )

    def on_model_prediction_ready(self, frames: List[VideoFrame]) -> None:
        for frame in frames:
            self._latency_monitors[frame.source_id].register_prediction_ready(
                frame_timestamp=frame.frame_timestamp,
                frame_id=frame.frame_id,
            )
            self._inference_throughput_monitor.tick()

    def get_report(self) -> PipelineStateReport:
        sources_metadata = []
        if self._video_sources is not None:
            sources_metadata = [s.describe_source() for s in self._video_sources]
        latency_reports = [
            monitor.summarise_reports() for monitor in self._latency_monitors.values()
        ]
        if hasattr(self._inference_throughput_monitor(), "fps"):
            _inference_throughput_fps = self._inference_throughput_monitor.fps
        else:
            _inference_throughput_fps = self._inference_throughput_monitor()
        return PipelineStateReport(
            video_source_status_updates=list(self._stream_updates),
            latency_reports=latency_reports,
            inference_throughput=_inference_throughput_fps,
            sources_metadata=sources_metadata,
        )