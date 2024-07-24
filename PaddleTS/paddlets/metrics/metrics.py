class MetricContainer(object):
    """Container holding a list of metrics.

    Args:
        metrics(List[str]|List[Metric]): List of metric or metric names.
        prefix(str): Prefix of metric names.

    Attributes:
        _prefix(str): Prefix of metric names.
        _metrics(List[Metric]): List of metric instance.
        _names(List[str]): List of metric names associated with eval_name.
    """

    def __init__(self, metrics: Union[List[str], List[Metric]],
                 prefix: str=""):
        self._prefix = prefix
        self._metrics = (metrics
                         if (metrics and isinstance(metrics[-1], Metric)) else
                         Metric.get_metrics_by_names(metrics))
        self._names = [prefix + metric._NAME for metric in self._metrics]

    def __call__(self, y_true: np.ndarray,
                 y_score: np.ndarray) -> Dict[str, float]:
        """Compute all metrics and store into a dict.

        Args:
            y_true(np.ndarray): Ground truth (correct) target values.
            y_score(np.ndarray): Estimated target values.

        Returns:
            Dict[str, float]: Dict of metrics.
        """
        logs = {}
        for metric in self._metrics:
            res = metric.metric_fn(y_true, y_score)
            logs[self._prefix + metric._NAME] = res
        return logs

