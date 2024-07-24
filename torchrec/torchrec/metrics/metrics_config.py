class RecMetricDef:
    """The dataclass that defines a RecMetric.

    Args:
        rec_tasks (List[RecTaskInfo]): this and next fields specify the RecTask
            information. ``rec_tasks`` specifies the RecTask information while
            ``rec_task_indices`` represents the indices that point to the
            RecTask information stored in the parent ``MetricsConfig``. Only one
            of the two fields should be specified.
        rec_task_indices (List[int]): see the doscstring of ``rec_tasks``.
        window_size (int): the window size for this metric. Note that this is global window size.
            The local window size is window_size / world_size, and must be larger than batch size.
        arguments (Optional[Dict[str, Any]]): any propritary arguments to be used
            by this Metric.
    """

    rec_tasks: List[RecTaskInfo] = field(default_factory=list)
    rec_task_indices: List[int] = field(default_factory=list)
    window_size: int = _DEFAULT_WINDOW_SIZE
    arguments: Optional[Dict[str, Any]] = None

class RecTaskInfo:
    name: str = "DefaultTask"
    label_name: str = "label"
    prediction_name: str = "prediction"
    weight_name: str = "weight"
    session_metric_def: Optional[SessionMetricDef] = (
        None  # used for session level metrics
    )
    is_negative_task: bool = False

