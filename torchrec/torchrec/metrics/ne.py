class NEMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.NE
    _computation_class: Type[RecMetricComputation] = NEMetricComputation

