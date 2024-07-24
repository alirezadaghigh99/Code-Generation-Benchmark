class MetricStream:
    def __init__(self, group_name: str, handler: MetricHandler):
        self.group_name = group_name
        self.handler = handler

    def add_value(self, metric_name: str, metric_value: int):
        self.handler.emit(
            MetricData(time.time(), self.group_name, metric_name, metric_value)
        )

