class RunInfo:
    """
    Containing data about compression of the model.
    """

    model: Optional[str] = None
    backend: Optional[BackendType] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_diff: Optional[float] = None
    compression_memory_usage: Optional[int] = None
    status: Optional[str] = None
    fps: Optional[float] = None
    time_total: Optional[float] = None
    time_compression: Optional[float] = None
    num_compress_nodes: Optional[NumCompressNodes] = None
    stats_from_output = StatsFromOutput()

    @staticmethod
    def format_time(time_elapsed):
        if time_elapsed is None:
            return None
        return str(timedelta(seconds=int(time_elapsed)))

    @staticmethod
    def format_memory_usage(memory):
        if memory is None:
            return None
        return int(memory)

    def get_result_dict(self):
        return {
            "Model": self.model,
            "Backend": self.backend.value if self.backend else None,
            "Metric name": self.metric_name,
            "Metric value": self.metric_value,
            "Metric diff": self.metric_diff,
            "Num FQ": self.num_compress_nodes.num_fq_nodes,
            "Num int4": self.num_compress_nodes.num_int4,
            "Num int8": self.num_compress_nodes.num_int8,
            "RAM MiB": self.format_memory_usage(self.compression_memory_usage),
            "Compr. time": self.format_time(self.time_compression),
            **self.stats_from_output.get_stats(),
            "Total time": self.format_time(self.time_total),
            "FPS": self.fps,
            "Status": self.status[:LIMIT_LENGTH_OF_STATUS] if self.status is not None else None,
        }

