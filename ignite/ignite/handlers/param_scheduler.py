def create_lr_scheduler_with_warmup(
    lr_scheduler: Union[ParamScheduler, PyTorchLRScheduler],
    warmup_start_value: float,
    warmup_duration: int,
    warmup_end_value: Optional[float] = None,
    save_history: bool = False,
    output_simulated_values: Optional[List] = None,
) -> "ConcatScheduler":
    """
    Helper method to create a learning rate scheduler with a linear warm-up.

    Args:
        lr_scheduler: learning rate scheduler after the warm-up.
        warmup_start_value: learning rate start value of the warm-up phase.
        warmup_duration: warm-up phase duration, number of events.
        warmup_end_value: learning rate end value of the warm-up phase, (default=None). If None,
             warmup_end_value is set to optimizer initial lr.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        output_simulated_values: optional output of simulated learning rate values.
            If output_simulated_values is a list of None, e.g. `[None] * 100`, after the execution it will be filled
            by 100 simulated learning rate values.

    Returns:
        ConcatScheduler

    Note:
        If the first learning rate value provided by `lr_scheduler` is different from `warmup_end_value`, an additional
        event is added after the warm-up phase such that the warm-up ends with `warmup_end_value` value and then
        `lr_scheduler` provides its learning rate values as normally.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            from torch.optim.lr_scheduler import ExponentialLR

            torch_lr_scheduler = ExponentialLR(optimizer=default_optimizer, gamma=0.98)

            default_trainer = get_default_trainer()

            scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                                        warmup_start_value=0.0,
                                                        warmup_end_value=0.1,
                                                        warmup_duration=3)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 8, max_epochs=1)

        .. testoutput::

            0.0
            0.05
            0.1
            0.098
            0.09604
            0.09411...
            0.09223...
            0.09039...

    .. versionadded:: 0.4.5
    """
    if not isinstance(lr_scheduler, (ParamScheduler, PyTorchLRScheduler)):
        raise TypeError(
            "Argument lr_scheduler should be a subclass of "
            f"torch.optim.lr_scheduler.{PyTorchLRScheduler.__name__} or ParamScheduler, "
            f"but given {type(lr_scheduler)}"
        )

    if not isinstance(warmup_duration, numbers.Integral):
        raise TypeError(f"Argument warmup_duration should be integer, but given {warmup_duration}")

    if not (warmup_duration > 1):
        raise ValueError(f"Argument warmup_duration should be at least 2 events, but given {warmup_duration}")

    warmup_schedulers: List[ParamScheduler] = []

    for param_group_index, param_group in enumerate(lr_scheduler.optimizer.param_groups):
        if warmup_end_value is None:
            param_group_warmup_end_value = param_group["lr"]
        else:
            param_group_warmup_end_value = warmup_end_value

        milestones_values = [(0, warmup_start_value), (warmup_duration - 1, param_group_warmup_end_value)]

        if isinstance(lr_scheduler, PyTorchLRScheduler):
            init_lr = param_group["lr"]
            if init_lr != param_group_warmup_end_value:
                milestones_values.append((warmup_duration, init_lr))

            # We need to advance torch lr_scheduler to avoid duplicated lr value
            # given by PiecewiseLinear and LRScheduler.
            # We suggest to attach output scheduler on ITERATION_STARTED but
            # torch lr_scheduler works with ITERATION_COMPLETED
            # See also https://github.com/pytorch/ignite/pull/2496#issuecomment-1065984440
            lr_scheduler.last_epoch += 1
            lr_scheduler = LRScheduler(lr_scheduler, save_history=save_history)
        else:
            init_lr = lr_scheduler.get_param()
            if init_lr == param_group_warmup_end_value:
                if warmup_duration > 2:
                    d = (param_group_warmup_end_value - warmup_start_value) / (warmup_duration - 1)
                    milestones_values[-1] = (warmup_duration - 2, param_group_warmup_end_value - d)
                else:
                    milestones_values.pop(-1)

        warmup_schedulers.append(
            PiecewiseLinear(
                lr_scheduler.optimizer,
                param_name="lr",
                milestones_values=milestones_values,
                param_group_index=param_group_index,
                save_history=save_history,
            )
        )

    warmup_scheduler = ParamGroupScheduler(warmup_schedulers, save_history=save_history)

    schedulers: List[Union[ParamScheduler, ParamGroupScheduler, PyTorchLRScheduler]] = [
        warmup_scheduler,
        lr_scheduler,
    ]
    durations = [milestones_values[-1][0] + 1]
    combined_scheduler = ConcatScheduler(schedulers, durations=durations, save_history=save_history)

    if output_simulated_values is not None:
        if not isinstance(output_simulated_values, list):
            raise TypeError(
                "Argument output_simulated_values should be a list of None, e.g. `[None] * 100`, "
                f"but given {type(output_simulated_values)}."
            )
        num_events = len(output_simulated_values)
        result = ConcatScheduler.simulate_values(num_events=num_events, schedulers=schedulers, durations=durations)
        for i in range(num_events):
            output_simulated_values[i] = result[i]
    return combined_scheduler

class LinearCyclicalScheduler(CyclicalScheduler):
    """Linearly adjusts param value to 'end_value' for a half-cycle, then linearly
    adjusts it back to 'start_value' for a half-cycle.

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: name of optimizer's parameter to update.
        start_value: value at start of cycle.
        end_value: value at the middle of the cycle.
        cycle_size: length of cycle.
        cycle_mult: ratio by which to change the cycle_size
            at the end of each cycle (default=1).
        start_value_mult: ratio by which to change the start value at the
            end of each cycle (default=1.0).
        end_value_mult: ratio by which to change the end value at the
            end of each cycle (default=1.0).
        warmup_duration: duration of warm-up to be applied before each cycle.
            Through this warm-up, the parameter starts from the last cycle's end value
            and linearly goes to next cycle's start value. Default is no cyclic warm-up.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        param_group_index: optimizer's parameters group to use.
        monotonic: whether to schedule only one half of the cycle: descending or ascending.
            If True, this argument can not be used together with ``warmup_duration``.
            (default=False).

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode:: 1

            default_trainer = get_default_trainer()

            # Linearly increases the learning rate from 0.0 to 1.0 and back to 0.0
            # over a cycle of 4 iterations
            scheduler = LinearCyclicalScheduler(default_optimizer, "lr", 0.0, 1.0, 4)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 9, max_epochs=1)

        .. testoutput:: 1

            0.0
            0.5
            1.0
            0.5
            ...

        .. testcode:: 2

            default_trainer = get_default_trainer()

            optimizer = torch.optim.SGD(
                [
                    {"params": default_model.base.parameters(), "lr": 0.001},
                    {"params": default_model.fc.parameters(), "lr": 0.01},
                ]
            )

            # Linearly increases the learning rate from 0.0 to 1.0 and back to 0.0
            # over a cycle of 4 iterations
            scheduler1 = LinearCyclicalScheduler(optimizer, "lr (base)", 0.0, 1.0, 4, param_group_index=0)

            # Linearly increases the learning rate from 0.0 to 0.1 and back to 0.0
            # over a cycle of 4 iterations
            scheduler2 = LinearCyclicalScheduler(optimizer, "lr (fc)", 0.0, 0.1, 4, param_group_index=1)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler1)
            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler2)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(optimizer.param_groups[0]["lr (base)"],
                      optimizer.param_groups[1]["lr (fc)"])

            default_trainer.run([0] * 9, max_epochs=1)

        .. testoutput:: 2

            0.0 0.0
            0.5 0.05
            1.0 0.1
            0.5 0.05
            ...

    .. versionadded:: 0.4.5

    .. versionchanged:: 0.4.13
        Added cyclic warm-up to the scheduler using ``warmup_duration``.

    .. versionchanged:: 0.5.0
        Added monotonic argument.
    """

    def __init__(self, *args: Any, monotonic: bool = False, **kwagrs: Any):
        super(LinearCyclicalScheduler, self).__init__(*args, **kwagrs)
        self.monotonic = monotonic
        if self.warmup_duration > 0 and not self.monotonic:
            raise ValueError(
                "Invalid combination when warmup_duration > 0 and monotonic=False, "
                "please use either set warmup_duration=0 or monotonic=True"
            )

    def get_param(self) -> float:
        """Method to get current optimizer's parameter value"""
        cycle_progress = self.event_index / self.cycle_size

        if self.monotonic:
            return self.start_value + (self.end_value - self.start_value) * cycle_progress
        else:
            return self.end_value + (self.start_value - self.end_value) * abs(cycle_progress - 0.5) * 2

class PiecewiseLinear(ParamScheduler):
    """
    Piecewise linear parameter scheduler

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: name of optimizer's parameter to update.
        milestones_values: list of tuples (event index, parameter value)
            represents milestones and parameter. Milestones should be increasing integers.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        param_group_index: optimizer's parameters group to use.

    .. code-block:: python

        scheduler = PiecewiseLinear(optimizer, "lr",
                                    milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)])
        # Attach to the trainer
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        #
        # Sets the learning rate to 0.5 over the first 10 iterations, then decreases linearly from 0.5 to 0.45 between
        # 10th and 20th iterations. Next there is a jump to 0.3 at the 21st iteration and LR decreases linearly
        # from 0.3 to 0.1 between 21st and 30th iterations and remains 0.1 until the end of the iterations.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode:: 1

            default_trainer = get_default_trainer()

            milestones_values = [(1, 1.0), (3, 0.8), (5, 0.2)]
            scheduler = PiecewiseLinear(
                default_optimizer, "lr", milestones_values=milestones_values)
            # Sets lr equal to 1 for till the first iteration
            # Then linearly reduces lr from 1 to 0.8 till the third iteration
            # Then linearly reduces lr from 0.8 to 0.5 till the fifth iteration

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 6, max_epochs=1)

        .. testoutput:: 1

            1.0
            1.0
            0.9
            0.8
            0.5
            0.2

        .. testcode:: 2

            default_trainer = get_default_trainer()

            optimizer = torch.optim.SGD(
                [
                    {"params": default_model.base.parameters(), "lr": 0.1},
                    {"params": default_model.fc.parameters(), "lr": 1.0},
                ]
            )

            milestones_values1 = [(1, 0.1), (3, 0.08), (5, 0.02)]
            scheduler2 = PiecewiseLinear(
                optimizer, "lr", milestones_values=milestones_values1, param_group_index=0)
            # Sets lr equal to 0.1 for till the first iteration
            # Then linearly reduces lr from 0.1 to 0.08 till the third iteration
            # Then linearly reduces lr from 0.08 to 0.05 till the fifth iteration

            milestones_values2 = [(1, 1.0), (3, 0.8), (5, 0.2)]
            scheduler1 = PiecewiseLinear(
                optimizer, "lr", milestones_values=milestones_values2, param_group_index=1)
            # Sets lr equal to 1 for till the first iteration
            # Then linearly reduces lr from 1 to 0.8 till the third iteration
            # Then linearly reduces lr from 0.8 to 0.5 till the fifth iteration

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler1)
            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler2)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(optimizer.param_groups[0]["lr"],
                      optimizer.param_groups[1]["lr"])

            default_trainer.run([0] * 6, max_epochs=1)

        .. testoutput:: 2

            0.1 1.0
            0.1 1.0
            0.09 0.9
            0.08 0.8
            0.05 0.5
            0.02 0.2

    .. versionadded:: 0.4.5
    """

    def __init__(
        self,
        optimizer: Optimizer,
        param_name: str,
        milestones_values: List[Tuple[int, float]],
        save_history: bool = False,
        param_group_index: Optional[int] = None,
    ):
        super(PiecewiseLinear, self).__init__(optimizer, param_name, save_history, param_group_index=param_group_index)

        if not isinstance(milestones_values, Sequence):
            raise TypeError(
                f"Argument milestones_values should be a list or tuple, but given {type(milestones_values)}"
            )
        if len(milestones_values) < 1:
            raise ValueError(
                f"Argument milestones_values should be with at least one value, but given {milestones_values}"
            )

        values: List[float] = []
        milestones: List[int] = []
        for pair in milestones_values:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Argument milestones_values should be a list of pairs (milestone, param_value)")
            if not isinstance(pair[0], numbers.Integral):
                raise TypeError(f"Value of a milestone should be integer, but given {type(pair[0])}")
            if len(milestones) > 0 and pair[0] < milestones[-1]:
                raise ValueError(
                    f"Milestones should be increasing integers, but given {pair[0]} is smaller "
                    f"than the previous milestone {milestones[-1]}"
                )
            milestones.append(pair[0])
            values.append(pair[1])

        self.values = values
        self.milestones = milestones
        self._index = 0
        self._state_attrs += ["values", "milestones", "_index"]

    def _get_start_end(self) -> Tuple[int, int, float, float]:
        if self.milestones[0] > self.event_index:
            return self.event_index - 1, self.event_index, self.values[0], self.values[0]
        elif self.milestones[-1] <= self.event_index:
            return (self.event_index, self.event_index + 1, self.values[-1], self.values[-1])
        elif self.milestones[self._index] <= self.event_index < self.milestones[self._index + 1]:
            return (
                self.milestones[self._index],
                self.milestones[self._index + 1],
                self.values[self._index],
                self.values[self._index + 1],
            )
        else:
            self._index += 1
            return self._get_start_end()

    def get_param(self) -> float:
        start_index, end_index, start_value, end_value = self._get_start_end()
        return start_value + (end_value - start_value) * (self.event_index - start_index) / (end_index - start_index)

class LRScheduler(ParamScheduler):
    """A wrapper class to call `torch.optim.lr_scheduler` objects as `ignite` handlers.

    Args:
        lr_scheduler: lr_scheduler object to wrap.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        use_legacy: if True, scheduler should be attached to ``Events.ITERATION_COMPLETED``, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            from torch.optim.lr_scheduler import StepLR

            torch_lr_scheduler = StepLR(default_optimizer, step_size=3, gamma=0.1)
            scheduler = LRScheduler(torch_lr_scheduler)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 8, max_epochs=1)

        .. testoutput::

            0.1
            0.1
            0.1
            0.010...
            0.010...
            0.010...
            0.001...
            0.001...

    .. versionadded:: 0.4.5

    ..  versionchanged:: 0.4.9
        added `use_legacy` argument
    """

    def __init__(
        self,
        lr_scheduler: PyTorchLRScheduler,
        save_history: bool = False,
        use_legacy: bool = False,
    ):
        if not isinstance(lr_scheduler, PyTorchLRScheduler):
            raise TypeError(
                "Argument lr_scheduler should be a subclass of "
                f"torch.optim.lr_scheduler.{PyTorchLRScheduler.__name__}, "
                f"but given {type(lr_scheduler)}"
            )

        self.lr_scheduler: Union[PyTorchLRScheduler, _CosineAnnealingWarmRestarts] = lr_scheduler
        if isinstance(lr_scheduler, CosineAnnealingWarmRestarts):
            self.lr_scheduler = _CosineAnnealingWarmRestarts(lr_scheduler)

        super(LRScheduler, self).__init__(
            optimizer=self.lr_scheduler.optimizer,
            param_name="lr",
            save_history=save_history,
        )
        if use_legacy:
            warnings.warn(
                "Please make sure to attach scheduler to Events.ITERATION_COMPLETED "
                "instead of Events.ITERATION_STARTED to make sure to use "
                "the first lr value from the optimizer, otherwise it will be skipped"
            )
            self.lr_scheduler.last_epoch += 1

        self._state_attrs += ["lr_scheduler"]

    def __call__(self, engine: Optional[Engine], name: Optional[str] = None) -> None:
        super(LRScheduler, self).__call__(engine, name)
        self.lr_scheduler.last_epoch += 1

    def get_param(self) -> Union[float, List[float]]:
        """Method to get current optimizer's parameter value"""
        # Emulate context manager for pytorch>=1.4
        self.lr_scheduler._get_lr_called_within_step = True  # type: ignore[union-attr]
        lr_list = self.lr_scheduler.get_lr()
        self.lr_scheduler._get_lr_called_within_step = False  # type: ignore[union-attr]
        if len(lr_list) == 1:
            return lr_list[0]
        else:
            return lr_list

    @classmethod
    def simulate_values(  # type: ignore[override]
        cls, num_events: int, lr_scheduler: PyTorchLRScheduler, **kwargs: Any
    ) -> List[List[int]]:
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events: number of events during the simulation.
            lr_scheduler: lr_scheduler object to wrap.

        Returns:
            event_index, value
        """

        if not isinstance(lr_scheduler, PyTorchLRScheduler):
            raise TypeError(
                "Argument lr_scheduler should be a subclass of "
                f"torch.optim.lr_scheduler.{PyTorchLRScheduler.__name__}, "
                f"but given {type(lr_scheduler)}"
            )

        # This scheduler uses `torch.optim.lr_scheduler.LRScheduler` which
        # should be replicated in order to simulate LR values and
        # not perturb original scheduler.
        with tempfile.TemporaryDirectory() as tmpdirname:
            cache_filepath = Path(tmpdirname) / "ignite_lr_scheduler_cache.pt"
            obj = {
                "lr_scheduler": lr_scheduler.state_dict(),
                "optimizer": lr_scheduler.optimizer.state_dict(),
            }
            torch.save(obj, cache_filepath.as_posix())

            values = []
            scheduler = cls(save_history=False, lr_scheduler=lr_scheduler, **kwargs)
            for i in range(num_events):
                scheduler(engine=None)
                params = [p[scheduler.param_name] for p in scheduler.optimizer_param_groups]
                values.append([i] + params)

            obj = torch.load(cache_filepath.as_posix())
            lr_scheduler.load_state_dict(obj["lr_scheduler"])
            lr_scheduler.optimizer.load_state_dict(obj["optimizer"])

            return values

class LRScheduler(ParamScheduler):
    """A wrapper class to call `torch.optim.lr_scheduler` objects as `ignite` handlers.

    Args:
        lr_scheduler: lr_scheduler object to wrap.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        use_legacy: if True, scheduler should be attached to ``Events.ITERATION_COMPLETED``, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            from torch.optim.lr_scheduler import StepLR

            torch_lr_scheduler = StepLR(default_optimizer, step_size=3, gamma=0.1)
            scheduler = LRScheduler(torch_lr_scheduler)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 8, max_epochs=1)

        .. testoutput::

            0.1
            0.1
            0.1
            0.010...
            0.010...
            0.010...
            0.001...
            0.001...

    .. versionadded:: 0.4.5

    ..  versionchanged:: 0.4.9
        added `use_legacy` argument
    """

    def __init__(
        self,
        lr_scheduler: PyTorchLRScheduler,
        save_history: bool = False,
        use_legacy: bool = False,
    ):
        if not isinstance(lr_scheduler, PyTorchLRScheduler):
            raise TypeError(
                "Argument lr_scheduler should be a subclass of "
                f"torch.optim.lr_scheduler.{PyTorchLRScheduler.__name__}, "
                f"but given {type(lr_scheduler)}"
            )

        self.lr_scheduler: Union[PyTorchLRScheduler, _CosineAnnealingWarmRestarts] = lr_scheduler
        if isinstance(lr_scheduler, CosineAnnealingWarmRestarts):
            self.lr_scheduler = _CosineAnnealingWarmRestarts(lr_scheduler)

        super(LRScheduler, self).__init__(
            optimizer=self.lr_scheduler.optimizer,
            param_name="lr",
            save_history=save_history,
        )
        if use_legacy:
            warnings.warn(
                "Please make sure to attach scheduler to Events.ITERATION_COMPLETED "
                "instead of Events.ITERATION_STARTED to make sure to use "
                "the first lr value from the optimizer, otherwise it will be skipped"
            )
            self.lr_scheduler.last_epoch += 1

        self._state_attrs += ["lr_scheduler"]

    def __call__(self, engine: Optional[Engine], name: Optional[str] = None) -> None:
        super(LRScheduler, self).__call__(engine, name)
        self.lr_scheduler.last_epoch += 1

    def get_param(self) -> Union[float, List[float]]:
        """Method to get current optimizer's parameter value"""
        # Emulate context manager for pytorch>=1.4
        self.lr_scheduler._get_lr_called_within_step = True  # type: ignore[union-attr]
        lr_list = self.lr_scheduler.get_lr()
        self.lr_scheduler._get_lr_called_within_step = False  # type: ignore[union-attr]
        if len(lr_list) == 1:
            return lr_list[0]
        else:
            return lr_list

    @classmethod
    def simulate_values(  # type: ignore[override]
        cls, num_events: int, lr_scheduler: PyTorchLRScheduler, **kwargs: Any
    ) -> List[List[int]]:
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events: number of events during the simulation.
            lr_scheduler: lr_scheduler object to wrap.

        Returns:
            event_index, value
        """

        if not isinstance(lr_scheduler, PyTorchLRScheduler):
            raise TypeError(
                "Argument lr_scheduler should be a subclass of "
                f"torch.optim.lr_scheduler.{PyTorchLRScheduler.__name__}, "
                f"but given {type(lr_scheduler)}"
            )

        # This scheduler uses `torch.optim.lr_scheduler.LRScheduler` which
        # should be replicated in order to simulate LR values and
        # not perturb original scheduler.
        with tempfile.TemporaryDirectory() as tmpdirname:
            cache_filepath = Path(tmpdirname) / "ignite_lr_scheduler_cache.pt"
            obj = {
                "lr_scheduler": lr_scheduler.state_dict(),
                "optimizer": lr_scheduler.optimizer.state_dict(),
            }
            torch.save(obj, cache_filepath.as_posix())

            values = []
            scheduler = cls(save_history=False, lr_scheduler=lr_scheduler, **kwargs)
            for i in range(num_events):
                scheduler(engine=None)
                params = [p[scheduler.param_name] for p in scheduler.optimizer_param_groups]
                values.append([i] + params)

            obj = torch.load(cache_filepath.as_posix())
            lr_scheduler.load_state_dict(obj["lr_scheduler"])
            lr_scheduler.optimizer.load_state_dict(obj["optimizer"])

            return values

class PiecewiseLinear(ParamScheduler):
    """
    Piecewise linear parameter scheduler

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: name of optimizer's parameter to update.
        milestones_values: list of tuples (event index, parameter value)
            represents milestones and parameter. Milestones should be increasing integers.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        param_group_index: optimizer's parameters group to use.

    .. code-block:: python

        scheduler = PiecewiseLinear(optimizer, "lr",
                                    milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)])
        # Attach to the trainer
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        #
        # Sets the learning rate to 0.5 over the first 10 iterations, then decreases linearly from 0.5 to 0.45 between
        # 10th and 20th iterations. Next there is a jump to 0.3 at the 21st iteration and LR decreases linearly
        # from 0.3 to 0.1 between 21st and 30th iterations and remains 0.1 until the end of the iterations.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode:: 1

            default_trainer = get_default_trainer()

            milestones_values = [(1, 1.0), (3, 0.8), (5, 0.2)]
            scheduler = PiecewiseLinear(
                default_optimizer, "lr", milestones_values=milestones_values)
            # Sets lr equal to 1 for till the first iteration
            # Then linearly reduces lr from 1 to 0.8 till the third iteration
            # Then linearly reduces lr from 0.8 to 0.5 till the fifth iteration

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 6, max_epochs=1)

        .. testoutput:: 1

            1.0
            1.0
            0.9
            0.8
            0.5
            0.2

        .. testcode:: 2

            default_trainer = get_default_trainer()

            optimizer = torch.optim.SGD(
                [
                    {"params": default_model.base.parameters(), "lr": 0.1},
                    {"params": default_model.fc.parameters(), "lr": 1.0},
                ]
            )

            milestones_values1 = [(1, 0.1), (3, 0.08), (5, 0.02)]
            scheduler2 = PiecewiseLinear(
                optimizer, "lr", milestones_values=milestones_values1, param_group_index=0)
            # Sets lr equal to 0.1 for till the first iteration
            # Then linearly reduces lr from 0.1 to 0.08 till the third iteration
            # Then linearly reduces lr from 0.08 to 0.05 till the fifth iteration

            milestones_values2 = [(1, 1.0), (3, 0.8), (5, 0.2)]
            scheduler1 = PiecewiseLinear(
                optimizer, "lr", milestones_values=milestones_values2, param_group_index=1)
            # Sets lr equal to 1 for till the first iteration
            # Then linearly reduces lr from 1 to 0.8 till the third iteration
            # Then linearly reduces lr from 0.8 to 0.5 till the fifth iteration

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler1)
            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler2)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(optimizer.param_groups[0]["lr"],
                      optimizer.param_groups[1]["lr"])

            default_trainer.run([0] * 6, max_epochs=1)

        .. testoutput:: 2

            0.1 1.0
            0.1 1.0
            0.09 0.9
            0.08 0.8
            0.05 0.5
            0.02 0.2

    .. versionadded:: 0.4.5
    """

    def __init__(
        self,
        optimizer: Optimizer,
        param_name: str,
        milestones_values: List[Tuple[int, float]],
        save_history: bool = False,
        param_group_index: Optional[int] = None,
    ):
        super(PiecewiseLinear, self).__init__(optimizer, param_name, save_history, param_group_index=param_group_index)

        if not isinstance(milestones_values, Sequence):
            raise TypeError(
                f"Argument milestones_values should be a list or tuple, but given {type(milestones_values)}"
            )
        if len(milestones_values) < 1:
            raise ValueError(
                f"Argument milestones_values should be with at least one value, but given {milestones_values}"
            )

        values: List[float] = []
        milestones: List[int] = []
        for pair in milestones_values:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Argument milestones_values should be a list of pairs (milestone, param_value)")
            if not isinstance(pair[0], numbers.Integral):
                raise TypeError(f"Value of a milestone should be integer, but given {type(pair[0])}")
            if len(milestones) > 0 and pair[0] < milestones[-1]:
                raise ValueError(
                    f"Milestones should be increasing integers, but given {pair[0]} is smaller "
                    f"than the previous milestone {milestones[-1]}"
                )
            milestones.append(pair[0])
            values.append(pair[1])

        self.values = values
        self.milestones = milestones
        self._index = 0
        self._state_attrs += ["values", "milestones", "_index"]

    def _get_start_end(self) -> Tuple[int, int, float, float]:
        if self.milestones[0] > self.event_index:
            return self.event_index - 1, self.event_index, self.values[0], self.values[0]
        elif self.milestones[-1] <= self.event_index:
            return (self.event_index, self.event_index + 1, self.values[-1], self.values[-1])
        elif self.milestones[self._index] <= self.event_index < self.milestones[self._index + 1]:
            return (
                self.milestones[self._index],
                self.milestones[self._index + 1],
                self.values[self._index],
                self.values[self._index + 1],
            )
        else:
            self._index += 1
            return self._get_start_end()

    def get_param(self) -> float:
        start_index, end_index, start_value, end_value = self._get_start_end()
        return start_value + (end_value - start_value) * (self.event_index - start_index) / (end_index - start_index)

