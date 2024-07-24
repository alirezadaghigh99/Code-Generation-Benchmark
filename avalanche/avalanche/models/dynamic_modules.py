class MultiHeadClassifier(MultiTaskModule):
    """Multi-head classifier with separate heads for each task.

    Typically used in task-incremental benchmarks where task labels are
    available and provided to the model.

    .. note::
        Each output head may have a different shape, and the number of
        classes can be determined automatically.

        However, since pytorch doest not support jagged tensors, when you
        compute a minibatch's output you must ensure that each sample
        has the same output size, otherwise the model will fail to
        concatenate the samples together.

        These can be easily ensured in two possible ways:

        - each minibatch contains a single task, which is the case in most
            common benchmarks in Avalanche. Some exceptions to this setting
            are multi-task replay or cumulative strategies.
        - each head has the same size, which can be enforced by setting a
            large enough `initial_out_features`.
    """

    def __init__(
        self,
        in_features,
        initial_out_features=2,
        masking=True,
        mask_value=-1000,
    ):
        """Init.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__()
        self.masking = masking
        self.mask_value = mask_value
        self.in_features = in_features
        self.starting_out_features = initial_out_features
        self.classifiers = torch.nn.ModuleDict()

        # needs to create the first head because pytorch optimizers
        # fail when model.parameters() is empty.
        # masking in IncrementalClassifier is unaware of task labels
        # so we do masking here instead.
        first_head = IncrementalClassifier(
            self.in_features,
            self.starting_out_features,
            masking=False,
            auto_adapt=False,
        )
        self.classifiers["0"] = first_head
        self.max_class_label = max(self.max_class_label, initial_out_features)

        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units_T0", au_init)

    @property
    def active_units(self):
        res = {}
        for tid in self.known_train_tasks_labels:
            mask = getattr(self, f"active_units_T{tid}").to(torch.bool)
            au = torch.arange(0, mask.shape[0])[mask].tolist()
            res[tid] = au
        return res

    @property
    def task_masks(self):
        res = {}
        for tid in self.known_train_tasks_labels:
            res[tid] = getattr(self, f"active_units_T{tid}").to(torch.bool)
        return res

    def adaptation(self, experience: CLExperience):
        """If `dataset` contains new tasks, a new head is initialized.

        :param experience: data from the current experience.
        :return:
        """
        super().adaptation(experience)
        device = self._adaptation_device
        curr_classes = experience.classes_in_this_experience
        task_labels = experience.task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            tid = str(tid)
            # head adaptation
            if tid not in self.classifiers:  # create new head
                new_head = IncrementalClassifier(
                    self.in_features,
                    self.starting_out_features,
                    masking=False,
                    auto_adapt=False,
                ).to(device)
                self.classifiers[tid] = new_head

                au_init = torch.zeros(
                    self.starting_out_features, dtype=torch.int8, device=device
                )
                self.register_buffer(f"active_units_T{tid}", au_init)

            self.classifiers[tid].adaptation(experience)

            # update active_units mask for the current task
            if self.masking:
                # TODO: code below assumes a single task for each experience
                # it should be easy to generalize but it may be slower.
                if len(task_labels) > 1:
                    raise NotImplementedError(
                        "Multi-Head unit masking is not supported when "
                        "experiences have multiple task labels. Set "
                        "masking=False in your "
                        "MultiHeadClassifier to disable masking."
                    )

                au_name = f"active_units_T{tid}"
                curr_head = self.classifiers[tid]
                old_nunits = self._buffers[au_name].shape[0]

                new_nclasses = max(
                    curr_head.classifier.out_features, max(curr_classes) + 1
                )
                if old_nunits != new_nclasses:  # expand active_units mask
                    old_act_units = self._buffers[au_name]
                    self._buffers[au_name] = torch.zeros(
                        new_nclasses, dtype=torch.int8, device=device
                    )
                    self._buffers[au_name][: old_act_units.shape[0]] = old_act_units
                # update with new active classes
                if self.training:
                    self._buffers[au_name][curr_classes] = 1

    def forward_single_task(self, x, task_label):
        """compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """
        device = self._adaptation_device
        task_label = str(task_label)
        out = self.classifiers[task_label](x)
        if self.masking:
            au_name = f"active_units_T{task_label}"
            curr_au = self._buffers[au_name]
            nunits, oldsize = out.shape[-1], curr_au.shape[0]
            if oldsize < nunits:  # we have to update the mask
                old_mask = self._buffers[au_name]
                self._buffers[au_name] = torch.zeros(
                    nunits, dtype=torch.int8, device=device
                )
                self._buffers[au_name][:oldsize] = old_mask
                curr_au = self._buffers[au_name]
            out[..., torch.logical_not(curr_au)] = self.mask_value
        return out

