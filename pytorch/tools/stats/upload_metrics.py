def add_global_metric(metric_name: str, metric_value: Any) -> None:
    """
    Adds stats that should be emitted with every metric by the current process.
    If the emit_metrics method specifies a metric with the same name, it will
    overwrite this value.
    """
    global_metrics[metric_name] = metric_value

