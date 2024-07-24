class RecMetric(nn.Module, abc.ABC):
    r"""The main class template to implement a recommendation metric.
    This class contains the recommendation tasks information (RecTaskInfo) and
    the actual computation object (RecMetricComputation). RecMetric processes
    all the information related to RecTaskInfo and models, and passes the required
    signals to the computation object, allowing the implementation of
    RecMetricComputation to focus on the mathematical meaning.

    A new metric that inherits RecMetric must override the following attributes
    in its own `__init__()`: `_namespace` and `_metrics_computations`. No other
    methods should be overridden.

    Args:
        world_size (int): the number of trainers.
        my_rank (int): the rank of this trainer.
        batch_size (int): batch size used by this trainer.
        tasks (List[RecTaskInfo]): the information of the model tasks.
        compute_mode (RecComputeMode): the computation mode. See RecComputeMode.
        window_size (int): the window size for the window metric.
        fused_update_limit (int): the maximum number of updates to be fused.
        compute_on_all_ranks (bool): whether to compute metrics on all ranks. This
            is necessary if the non-leader rank wants to consume global metrics result.
        should_validate_update (bool): whether to check the inputs of `update()` and
            skip the update if the inputs are invalid. Invalid inputs include the case
            where all examples have 0 weights for a batch.
        process_group (Optional[ProcessGroup]): the process group used for the
            communication. Will use the default process group if not specified.

    Example::

        ne = NEMetric(
            world_size=4,
            my_rank=0,
            batch_size=128,
            tasks=DefaultTaskInfo,
        )
    """

    _computation_class: Type[RecMetricComputation]
    _namespace: MetricNamespaceBase
    _metrics_computations: nn.ModuleList

    _tasks: List[RecTaskInfo]
    _window_size: int
    _tasks_iter: Callable[[str], ComputeIterType]
    _update_buffers: Dict[str, List[RecModelOutput]]
    _default_weights: Dict[Tuple[int, ...], torch.Tensor]

    _required_inputs: Set[str]

    PREDICTIONS: str = "predictions"
    LABELS: str = "labels"
    WEIGHTS: str = "weights"

    def __init__(
        self,
        world_size: int,
        my_rank: int,
        batch_size: int,
        tasks: List[RecTaskInfo],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        window_size: int = 100,
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        torch._C._log_api_usage_once(
            f"torchrec.metrics.rec_metric.{self.__class__.__name__}"
        )
        # TODO(stellaya): consider to inherit from TorchMetrics.Metric or
        # TorchMetrics.MetricCollection.
        if (
            compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION
            and fused_update_limit > 0
        ):
            raise ValueError(
                "The fused tasks computation and the fused update cannot be set at the same time"
            )
        super().__init__()
        self._world_size = world_size
        self._my_rank = my_rank
        self._window_size = math.ceil(window_size / world_size)
        self._batch_size = batch_size
        self._metrics_computations = nn.ModuleList()
        self._tasks = tasks
        self._compute_mode = compute_mode
        self._fused_update_limit = fused_update_limit
        self._should_validate_update = should_validate_update
        self._default_weights = {}
        self._required_inputs = set()
        self._update_buffers = {
            self.PREDICTIONS: [],
            self.LABELS: [],
            self.WEIGHTS: [],
        }
        if self._window_size < self._batch_size:
            raise ValueError(
                f"Local window size must be larger than batch size. Got local window size {self._window_size} and batch size {self._batch_size}."
            )

        if compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION:
            task_per_metric = len(self._tasks)
            self._tasks_iter = self._fused_tasks_iter
        else:
            task_per_metric = 1
            self._tasks_iter = self._unfused_tasks_iter

        for task_config in (
            [self._tasks]
            if compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION
            else self._tasks
        ):
            # pyre-ignore
            kwargs["fused_update_limit"] = fused_update_limit
            # This Pyre error seems to be Pyre's bug as it can be inferred by mypy
            # according to https://github.com/python/mypy/issues/3048.
            # pyre-fixme[45]: Cannot instantiate abstract class `RecMetricCoputation`.
            metric_computation = self._computation_class(
                my_rank,
                batch_size,
                task_per_metric,
                self._window_size,
                compute_on_all_ranks,
                self._should_validate_update,
                process_group,
                **{**kwargs, **self._get_task_kwargs(task_config)},
            )
            required_inputs = self._get_task_required_inputs(task_config)

            self._metrics_computations.append(metric_computation)
            self._required_inputs.update(required_inputs)

    def _get_task_kwargs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Dict[str, Any]:
        return {}

    def _get_task_required_inputs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Set[str]:
        return set()

    # TODO(stellaya): Refactor the _[fused, unfused]_tasks_iter methods and replace the
    # compute_scope str input with an enum
    def _fused_tasks_iter(self, compute_scope: str) -> ComputeIterType:
        assert len(self._metrics_computations) == 1
        self._metrics_computations[0].pre_compute()
        for metric_report in getattr(
            self._metrics_computations[0], compute_scope + "compute"
        )():
            for task, metric_value, has_valid_update in zip(
                self._tasks,
                metric_report.value,
                (
                    self._metrics_computations[0].has_valid_update
                    if self._should_validate_update
                    else itertools.repeat(1)
                ),  # has_valid_update > 0 means the update is valid
            ):
                # The attribute has_valid_update is a tensor whose length equals to the
                # number of tasks. Each value in it is corresponding to whether a task
                # has valid updates or not.
                # If for a task there's no valid updates, the calculated metric_value
                # will be meaningless, so we mask it with the default value, i.e. 0.
                valid_metric_value = (
                    metric_value
                    if has_valid_update > 0
                    else torch.zeros_like(metric_value)
                )
                yield task, metric_report.name, valid_metric_value, compute_scope + metric_report.metric_prefix.value, metric_report.description

    def _unfused_tasks_iter(self, compute_scope: str) -> ComputeIterType:
        for task, metric_computation in zip(self._tasks, self._metrics_computations):
            metric_computation.pre_compute()
            for metric_report in getattr(
                metric_computation, compute_scope + "compute"
            )():
                # The attribute has_valid_update is a tensor with only 1 value
                # corresponding to whether the task has valid updates or not.
                # If there's no valid update, the calculated metric_report.value
                # will be meaningless, so we mask it with the default value, i.e. 0.
                valid_metric_value = (
                    metric_report.value
                    if not self._should_validate_update
                    or metric_computation.has_valid_update[0] > 0
                    else torch.zeros_like(metric_report.value)
                )
                yield task, metric_report.name, valid_metric_value, compute_scope + metric_report.metric_prefix.value, metric_report.description

    def _fuse_update_buffers(self) -> Dict[str, RecModelOutput]:
        def fuse(outputs: List[RecModelOutput]) -> RecModelOutput:
            assert len(outputs) > 0
            if isinstance(outputs[0], torch.Tensor):
                return torch.cat(cast(List[torch.Tensor], outputs))
            else:
                task_outputs: Dict[str, List[torch.Tensor]] = defaultdict(list)
                for output in outputs:
                    assert isinstance(output, dict)
                    for task_name, tensor in output.items():
                        task_outputs[task_name].append(tensor)
                return {
                    name: torch.cat(tensors) for name, tensors in task_outputs.items()
                }

        ret: Dict[str, RecModelOutput] = {}
        for key, output_list in self._update_buffers.items():
            if len(output_list) > 0:
                ret[key] = fuse(output_list)
            else:
                assert key == self.WEIGHTS
            output_list.clear()
        return ret

    def _check_fused_update(self, force: bool) -> None:
        if self._fused_update_limit <= 0:
            return
        if len(self._update_buffers[self.PREDICTIONS]) == 0:
            return
        if (
            not force
            and len(self._update_buffers[self.PREDICTIONS]) < self._fused_update_limit
        ):
            return
        fused_arguments = self._fuse_update_buffers()
        self._update(
            predictions=fused_arguments[self.PREDICTIONS],
            labels=fused_arguments[self.LABELS],
            weights=fused_arguments.get(self.WEIGHTS, None),
        )

    def _create_default_weights(self, predictions: torch.Tensor) -> torch.Tensor:
        # pyre-fixme[6]: For 1st param expected `Tuple[int, ...]` but got `Size`.
        weights = self._default_weights.get(predictions.size(), None)
        if weights is None:
            weights = torch.ones_like(predictions)
            # pyre-fixme[6]: For 1st param expected `Tuple[int, ...]` but got `Size`.
            self._default_weights[predictions.size()] = weights
        return weights

    def _check_nonempty_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return torch.gt(torch.count_nonzero(weights, dim=-1), 0)

    def _update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
        **kwargs: Dict[str, Any],
    ) -> None:
        with torch.no_grad():
            if self._compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION:
                assert isinstance(predictions, torch.Tensor) and isinstance(
                    labels, torch.Tensor
                )

                predictions = (
                    # Reshape the predictions to size([len(self._tasks), self._batch_size])
                    predictions.view(-1, self._batch_size)
                    if predictions.dim() == labels.dim()
                    # predictions.dim() == labels.dim() + 1 for multiclass models
                    else predictions.view(-1, self._batch_size, predictions.size()[-1])
                )
                labels = labels.view(-1, self._batch_size)
                if weights is None:
                    weights = self._create_default_weights(predictions)
                else:
                    assert isinstance(weights, torch.Tensor)
                    weights = weights.view(-1, self._batch_size)
                if self._should_validate_update:
                    # has_valid_weights is a tensor of bool whose length equals to the number
                    # of tasks. Each value in it is corresponding to whether the weights
                    # are valid, i.e. are set to non-zero values for that task in this update.
                    # If has_valid_weights are Falses for all the tasks, we just ignore this
                    # update.
                    has_valid_weights = self._check_nonempty_weights(weights)
                    if torch.any(has_valid_weights):
                        self._metrics_computations[0].update(
                            predictions=predictions,
                            labels=labels,
                            weights=weights,
                            **kwargs,
                        )
                        self._metrics_computations[0].has_valid_update.logical_or_(
                            has_valid_weights
                        )
                else:
                    self._metrics_computations[0].update(
                        predictions=predictions,
                        labels=labels,
                        weights=weights,
                        **kwargs,
                    )
            else:
                for task, metric_ in zip(self._tasks, self._metrics_computations):
                    if task.name not in predictions:
                        continue
                    # pyre-fixme[6]: For 1st argument expected `Union[None,
                    #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any,
                    #  ...]]` but got `str`.
                    if torch.numel(predictions[task.name]) == 0:
                        # pyre-fixme[6]: For 1st argument expected `Union[None,
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        assert torch.numel(labels[task.name]) == 0
                        # pyre-fixme[6]: For 1st argument expected `Union[None,
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        assert weights is None or torch.numel(weights[task.name]) == 0
                        continue
                    task_predictions = (
                        # pyre-fixme[6]: For 1st argument expected `Union[None,
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        predictions[task.name].view(1, -1)
                        # pyre-fixme[6]: For 1st argument expected `Union[None,
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        if predictions[task.name].dim() == labels[task.name].dim()
                        # predictions[task.name].dim() == labels[task.name].dim() + 1 for multiclass models
                        # pyre-fixme[6]: For 1st argument expected `Union[None,
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        else predictions[task.name].view(
                            1,
                            -1,
                            predictions[
                                task.name  # pyre-fixme[6]: For 1st argument expected `Union[None,
                                #  List[typing.Any], int, slice, Tensor,
                                #  typing.Tuple[typing.Any, ...]]` but got `str`.
                            ].size()[-1],
                        )
                    )
                    # pyre-fixme[6]: For 1st argument expected `Union[None,
                    #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any,
                    #  ...]]` but got `str`.
                    task_labels = labels[task.name].view(1, -1)
                    if weights is None:
                        task_weights = self._create_default_weights(task_predictions)
                    else:
                        # pyre-fixme[6]: For 1st argument expected `Union[None,
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        task_weights = weights[task.name].view(1, -1)
                    if self._should_validate_update:
                        # has_valid_weights is a tensor with only 1 value corresponding to
                        # whether the weights are valid, i.e. are set to non-zero values for
                        # the task in this update.
                        # If has_valid_update[0] is False, we just ignore this update.
                        has_valid_weights = self._check_nonempty_weights(task_weights)
                        if has_valid_weights[0]:
                            metric_.has_valid_update.logical_or_(has_valid_weights)
                        else:
                            continue
                    if "required_inputs" in kwargs:
                        kwargs["required_inputs"] = {
                            k: v.view(task_labels.size())
                            for k, v in kwargs["required_inputs"].items()
                        }
                    metric_.update(
                        predictions=task_predictions,
                        labels=task_labels,
                        weights=task_weights,
                        **kwargs,
                    )

    def update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
        **kwargs: Dict[str, Any],
    ) -> None:
        with record_function(f"## {self.__class__.__name__}:update ##"):
            if self._fused_update_limit > 0:
                self._update_buffers[self.PREDICTIONS].append(predictions)
                self._update_buffers[self.LABELS].append(labels)
                if weights is not None:
                    self._update_buffers[self.WEIGHTS].append(weights)
                self._check_fused_update(force=False)
            else:
                self._update(
                    predictions=predictions, labels=labels, weights=weights, **kwargs
                )

    # The implementation of compute is very similar to local_compute, but compute overwrites
    # the abstract method compute in torchmetrics.Metric, which is wrapped by _wrap_compute
    def compute(self) -> Dict[str, torch.Tensor]:
        self._check_fused_update(force=True)
        ret = {}
        for task, metric_name, metric_value, prefix, description in self._tasks_iter(
            ""
        ):
            metric_key = compose_metric_key(
                self._namespace, task.name, metric_name, prefix, description
            )
            ret[metric_key] = metric_value
        return ret

    def local_compute(self) -> Dict[str, torch.Tensor]:
        self._check_fused_update(force=True)
        ret = {}
        for task, metric_name, metric_value, prefix, description in self._tasks_iter(
            "local_"
        ):
            metric_key = compose_metric_key(
                self._namespace, task.name, metric_name, prefix, description
            )
            ret[metric_key] = metric_value
        return ret

    def sync(self) -> None:
        for computation in self._metrics_computations:
            computation.sync()

    def unsync(self) -> None:
        for computation in self._metrics_computations:
            if computation._is_synced:
                computation.unsync()

    def reset(self) -> None:
        for computation in self._metrics_computations:
            computation.reset()

    def get_memory_usage(self) -> Dict[torch.Tensor, int]:
        r"""Estimates the memory of the rec metric instance's
        underlying tensors; returns the map of tensor to size
        """
        tensor_map = {}
        attributes_q = deque(self.__dict__.values())
        while attributes_q:
            attribute = attributes_q.popleft()
            if isinstance(attribute, torch.Tensor):
                tensor_map[attribute] = get_tensor_size_bytes(attribute)
            elif isinstance(attribute, WindowBuffer):
                attributes_q.extend(attribute.buffers)
            elif isinstance(attribute, Mapping):
                attributes_q.extend(attribute.values())
            elif isinstance(attribute, Sequence) and not isinstance(attribute, str):
                attributes_q.extend(attribute)
            elif hasattr(attribute, "__dict__") and not isinstance(attribute, Enum):
                attributes_q.extend(attribute.__dict__.values())
        return tensor_map

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # We need to flush the cached output to ensure checkpointing correctness.
        self._check_fused_update(force=True)
        destination = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return self._metrics_computations.state_dict(
            destination=destination,
            prefix=f"{prefix}_metrics_computations.",
            keep_vars=keep_vars,
        )

    def get_required_inputs(self) -> Set[str]:
        return self._required_inputs

