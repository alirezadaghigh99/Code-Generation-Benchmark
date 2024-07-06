def compose_metric_key(
    namespace: MetricNamespaceBase,
    task_name: str,
    metric_name: MetricNameBase,
    metric_prefix: MetricPrefix = MetricPrefix.DEFAULT,
    description: Optional[str] = None,
) -> str:
    r"""Get the metric key based on the input parameters"""
    return compose_customized_metric_key(
        compose_metric_namespace(namespace, task_name),
        f"{metric_prefix}{metric_name}",
        description,
    )

