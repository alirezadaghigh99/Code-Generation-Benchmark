def _make_taskaware_classification_dataset(
    dataset: TaskAwareSupervisedClassificationDataset,
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, TransformGroupDef]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset: ...

def _taskaware_classification_subset(
    dataset: TaskAwareSupervisedClassificationDataset,
    indices: Optional[Sequence[int]] = None,
    *,
    class_mapping: Optional[Sequence[int]] = None,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Mapping[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset: ...

def _make_taskaware_tensor_classification_dataset(
    *dataset_tensors: Sequence,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Dict[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = "train",
    task_labels: Union[int, Sequence[int]],
    targets: Union[Sequence[TTargetType], int],
    collate_fn: Optional[Callable[[List], Any]] = None
) -> TaskAwareSupervisedClassificationDataset: ...

