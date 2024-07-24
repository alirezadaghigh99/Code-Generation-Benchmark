class ParametricBuffer(BalancedExemplarsBuffer):
    """Stores samples for replay using a custom selection strategy and
    grouping."""

    def __init__(
        self,
        max_size: int,
        groupby=None,
        selection_strategy: Optional["ExemplarsSelectionStrategy"] = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param groupby: Grouping mechanism. One of {None, 'class', 'task',
            'experience'}.
        :param selection_strategy: The strategy used to select exemplars to
            keep in memory when cutting it off.
        """
        super().__init__(max_size)
        assert groupby in {None, "task", "class", "experience"}, (
            "Unknown grouping scheme. Must be one of {None, 'task', "
            "'class', 'experience'}"
        )
        self.groupby = groupby
        ss = selection_strategy or RandomExemplarsSelectionStrategy()
        self.selection_strategy = ss
        self.seen_groups: Set[int] = set()
        self._curr_strategy = None

    def post_adapt(self, agent, exp):
        new_data: AvalancheDataset = exp.dataset
        new_groups = self._make_groups(agent, new_data)
        self.seen_groups.update(new_groups.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_groups))
        group_to_len = {}
        for group_id, ll in zip(self.seen_groups, lens):
            group_to_len[group_id] = ll

        # update buffers with new data
        for group_id, new_data_g in new_groups.items():
            ll = group_to_len[group_id]
            if group_id in self.buffer_groups:
                old_buffer_g = self.buffer_groups[group_id]
                old_buffer_g.update_from_dataset(agent, new_data_g)
                old_buffer_g.resize(agent, ll)
            else:
                new_buffer = _ParametricSingleBuffer(ll, self.selection_strategy)
                new_buffer.update_from_dataset(agent, new_data_g)
                self.buffer_groups[group_id] = new_buffer

        # resize buffers
        for group_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[group_id].resize(agent, group_to_len[group_id])

    def _make_groups(
        self, strategy, data: AvalancheDataset
    ) -> Dict[int, AvalancheDataset]:
        """Split the data by group according to `self.groupby`."""
        if self.groupby is None:
            return {0: data}
        elif self.groupby == "task":
            return self._split_by_task(data)
        elif self.groupby == "experience":
            return self._split_by_experience(strategy, data)
        elif self.groupby == "class":
            return self._split_by_class(data)
        else:
            assert False, "Invalid groupby key. Should never get here."

    def _split_by_class(self, data: AvalancheDataset) -> Dict[int, AvalancheDataset]:
        # Get sample idxs per class
        cl_idxs: Dict[int, List[int]] = defaultdict(list)
        targets = getattr(data, "targets")
        for idx, target in enumerate(targets):
            target = int(target)
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        new_groups: Dict[int, AvalancheDataset] = {}
        for c, c_idxs in cl_idxs.items():
            new_groups[c] = _taskaware_classification_subset(data, indices=c_idxs)
        return new_groups

    def _split_by_experience(
        self, strategy, data: AvalancheDataset
    ) -> Dict[int, AvalancheDataset]:
        exp_id = strategy.clock.train_exp_counter + 1
        return {exp_id: data}

    def _split_by_task(self, data: AvalancheDataset) -> Dict[int, AvalancheDataset]:
        new_groups = {}
        task_set = getattr(data, "task_set")
        for task_id in task_set:
            new_groups[task_id] = task_set[task_id]
        return new_groups

class ClassBalancedBuffer(BalancedExemplarsBuffer[ReservoirSamplingBuffer]):
    """Stores samples for replay, equally divided over classes.

    There is a separate buffer updated by reservoir sampling for each class.
    It should be called in the 'after_training_exp' phase (see
    ExperienceBalancedStoragePolicy).
    The number of classes can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed classes so far.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: Optional[int] = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert total_num_classes is not None and (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes: Set[int] = set()

    def post_adapt(self, agent, exp):
        """Update buffer."""
        self.update_from_dataset(exp.dataset, agent)

    def update_from_dataset(
        self, new_data: AvalancheDataset, strategy: Optional["BaseSGDTemplate"] = None
    ):
        if len(new_data) == 0:
            return

        targets = getattr(new_data, "targets", None)
        assert targets is not None

        # Get sample idxs per class
        cl_idxs: Dict[int, List[int]] = defaultdict(list)
        for idx, target in enumerate(targets):
            # Conversion to int may fix issues when target
            # is a single-element torch.tensor
            target = int(target)
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = _taskaware_classification_subset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy, class_to_len[class_id])

class ExperienceBalancedBuffer(BalancedExemplarsBuffer[ReservoirSamplingBuffer]):
    """Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(self, max_size: int, adaptive_size: bool = True, num_experiences=None):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self._num_exps = 0

    def post_adapt(self, agent, exp):
        self._num_exps += 1
        new_data = exp.dataset
        lens = self.get_group_lengths(self._num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1])
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[self._num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(agent, ll)

