class TaskBalancedDataLoader(GroupBalancedDataLoader):
    """Task-balanced data loader for Avalanche's datasets."""

    def __init__(
        self,
        data: AvalancheDataset,
        batch_size: int = 32,
        oversample_small_groups: bool = False,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Task-balanced data loader for Avalanche's datasets.

        The iterator returns a mini-batch balanced across each task, which
        makes it useful when training in multi-task scenarios whenever data is
        highly unbalanced.

        If `oversample_small_tasks == True` smaller tasks are
        oversampled to match the largest task. Otherwise, once the data for a
        specific task is terminated, that task will not be present in the
        subsequent mini-batches.

        :param data: an instance of `AvalancheDataset`.
        :param oversample_small_groups: whether smaller tasks should be
            oversampled to match the largest one.
        :param distributed_sampling: If True, apply the PyTorch
            :class:`DistributedSampler`. Defaults to True.
            Note: the distributed sampler is not applied if not running
            a distributed training, even when True is passed.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        if "oversample_small_tasks" in kwargs:
            raise ValueError(
                "oversample_small_tasks is deprecated in favor of "
                "oversample_small_groups"
            )

        # Split data by task
        task_datasets = []
        task_labels_field = getattr(data, "targets_task_labels")
        assert isinstance(task_labels_field, DataAttribute)
        for task_label in task_labels_field.uniques:
            tidxs = task_labels_field.val_to_idx[task_label]
            tdata = data.subset(tidxs)
            task_datasets.append(tdata)

        super().__init__(
            task_datasets,
            oversample_small_groups=oversample_small_groups,
            batch_size=batch_size,
            distributed_sampling=distributed_sampling,
            **kwargs
        )

class ReplayDataLoader(MultiDatasetDataLoader):
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(
        self,
        data: AvalancheDataset,
        memory: Optional[AvalancheDataset] = None,
        oversample_small_tasks: bool = False,
        batch_size: int = 32,
        batch_size_mem: int = 32,
        task_balanced_dataloader: bool = False,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Custom data loader for rehearsal strategies.

        This dataloader iterates in parallel two datasets, the current `data`
        and the rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).

        The length of the loader is determined only by the current
        task data and is the same than what it would be when creating a
        data loader for this dataset.

        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param distributed_sampling: If True, apply the PyTorch
            :class:`DistributedSampler`. Defaults to True.
            Note: the distributed sampler is not applied if not running
            a distributed training, even when True is passed.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = data.collate_fn

        # Create dataloader for memory items
        if task_balanced_dataloader:
            memory_task_labels = getattr(memory, "targets_task_labels")
            assert isinstance(memory_task_labels, DataAttribute)
            num_keys = len(memory_task_labels.uniques)

            # Ensure that the per-task batch size will end up > 0
            assert batch_size_mem >= num_keys, (
                "Batch size must be greator or equal "
                "to the number of tasks in the memory "
                "and current data."
            )

            # Make the batch size balanced between tasks
            # The remainder (remaining_example) will be distributed
            # across tasks by "self._get_datasets_and_batch_sizes(...)"
            single_group_batch_size = batch_size_mem // num_keys
            remaining_example = batch_size_mem % num_keys
        else:
            single_group_batch_size = batch_size_mem
            remaining_example = 0

        # For current data, use the batch_size from the input "batch_size".
        # batch_size can be an int (do not split by task)
        # or a dictionary task_id -> mb_size
        # In both cases, remaining_examples=0
        data_batch_sizes, data_subsets = self._get_datasets_and_batch_sizes(
            data, batch_size, 0, False
        )

        memory_batch_sizes, memory_subsets = self._get_datasets_and_batch_sizes(
            memory,
            single_group_batch_size,
            remaining_example,
            task_balanced_dataloader,
        )

        # Obtain the subset with the highest number of iterations
        # This is the one that defines when an epoch ends
        # Note: this is aligned with the behavior of the legacy
        # multi-loader version of ReplayDataLoader
        loaders_for_len_estimation = []

        for data_subset, subset_mb_size in zip(data_subsets, data_batch_sizes):
            loaders_for_len_estimation.append(
                _make_data_loader(
                    data_subset,
                    distributed_sampling,
                    kwargs,
                    subset_mb_size,
                    force_no_workers=True,
                )[0]
            )

        longest_data_subset_idx = (
            np.array(len(d) for d in loaders_for_len_estimation).argmax().item()
        )

        super().__init__(
            data_subsets + memory_subsets,
            data_batch_sizes + memory_batch_sizes,
            termination_dataset=longest_data_subset_idx,
            oversample_small_datasets=oversample_small_tasks,
            distributed_sampling=distributed_sampling,
            **kwargs
        )

    @staticmethod
    def _get_datasets_and_batch_sizes(
        data: AvalancheDataset,
        batch_sizes_def: Union[int, Dict[int, int]],
        remaining_examples: int,
        task_balanced_dataloader: bool,
    ):
        datasets: List[AvalancheDataset] = []
        batch_sizes: List[int] = []
        batch_size_per_task = not isinstance(batch_sizes_def, int)

        if task_balanced_dataloader or batch_size_per_task:
            for task_id in data.task_set:
                dataset = data.task_set[task_id]

                if batch_size_per_task:
                    current_batch_size = batch_sizes_def[task_id]
                else:
                    current_batch_size = batch_sizes_def

                if remaining_examples > 0:
                    current_batch_size += 1
                    remaining_examples -= 1

                datasets.append(dataset)
                batch_sizes.append(current_batch_size)
        else:
            # Current data is loaded without task balancing
            datasets.append(data)
            batch_sizes.append(batch_sizes_def)
        return batch_sizes, datasets

