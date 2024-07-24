class StepParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the parameter value from outside this scheduler.

    Args:
        optimizer (BaseOptimWrapper or Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        step_size (int): Period of parameter value decay.
        gamma (float): Multiplicative factor of parameter value decay.
            Defaults to 0.1.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 step_size: int,
                 gamma: float = 0.1,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(
            optimizer=optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              step_size,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        step_size = step_size * epoch_length
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(
            *args,
            step_size=step_size,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if (self.last_step == 0) or (self.last_step % self.step_size != 0):
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        return [
            group[self.param_name] * self.gamma
            for group in self.optimizer.param_groups
        ]

class _ParamScheduler:
    """Base class for parameter schedulers.

    It should be inherited by all schedulers that schedule parameters in the
    optimizer's ``param_groups``. All subclasses should overwrite the
    ``_get_value()`` according to their own schedule strategy.
    The implementation is motivated by
    https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py.

    Args:
        optimizer (BaseOptimWrapper or Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resuming without
            state dict. Default value ``-1`` means the ``step`` function is
            never be called before. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """  # noqa: E501

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, (Optimizer, BaseOptimWrapper)):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))
        self.base_values = [
            group[f'initial_{param_name}'] for group in optimizer.param_groups
        ]
        self.last_step = last_step

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method: Callable):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)  # type: ignore
            # Get the unbound method for the same purpose.
            func = method.__func__  # type: ignore
            cls = instance_ref().__class__  # type: ignore
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._global_step += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore
            return wrapper

        # add counter to optimizer
        self.optimizer.step = with_counter(self.optimizer.step)  # type: ignore
        self.optimizer._global_step = -1  # type: ignore

        self._global_step = -1
        self.verbose = verbose

        self.step()

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not
        the optimizer.

        Returns:
            dict: scheduler state.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """Return the last computed value by current scheduler.

        Returns:
            list: A list of the last computed value of the optimizer's
            ``param_group``.
        """
        return self._last_value

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        raise NotImplementedError

    def print_value(self, is_verbose: bool, group: int, value: float):
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            print_log(
                f'Adjusting parameter value of group {group} to {value:.4e}.',
                logger='current')

    def step(self):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._global_step == 0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after '
                    'parameter value scheduler initialization. Please, make '
                    'sure to call `optimizer.step()` before '
                    '`scheduler.step()`. See more details at '
                    'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)

            # Just check if there were two first scheduler.step() calls
            # before optimizer.step()
            elif self.optimizer._global_step < 0:
                warnings.warn(
                    'Detected call of `scheduler.step()` before '
                    '`optimizer.step()`. In PyTorch 1.1.0 and later, you '
                    'should call them in the opposite order: '
                    '`optimizer.step()` before `scheduler.step()`. '
                    'Failure to do this will result in PyTorch skipping '
                    'the first value of the parameter value schedule. '
                    'See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)
        self._global_step += 1

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1
            values = self._get_value()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value = data
                param_group[self.param_name] = value
                self.print_value(self.verbose, i, value)

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]

class OneCycleParamScheduler(_ParamScheduler):
    r"""Sets the parameters of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every
    batch. `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in
    one of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. If total_steps is not defined, begin and end of the ParamSchedul will
       works for it. In this case, the number of total steps is inferred by
       total_steps = end - begin

    The default behaviour of this scheduler follows the fastai implementation
    of 1cycle, which claims that "unpublished work has shown even better
    results by using only two phases". To mimic the behaviour of the original
    paper instead, set ``three_phase=True``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        eta_max (float or list): Upper parameter value boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it will be equal to
            ``end - begin``. Defaults to None
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Defaults to 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing,
            "linear" for linear annealing.
            Defaults to 'cos'
        div_factor (float): Determines the initial learning rate via
            initial_param = eta_max/div_factor
            Defaults to 25
        final_div_factor (float): Determines the minimum learning rate via
            eta_min = initial_param/final_div_factor
            Defaults to 1e4
        three_phase (bool): If ``True``, use a third phase of the schedule to
            annihilate the learning rate according to 'final_div_factor'
            instead of modifying the second phase (the first two phases will be
            symmetrical about the step indicated by 'pct_start').
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.

    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """  # noqa E501

    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 eta_max: float = 0,
                 total_steps: Optional[int] = None,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 div_factor: float = 25.,
                 final_div_factor: float = 1e4,
                 three_phase: bool = False,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        assert param_name == 'lr', ('OneCycle only works for learning rate '
                                    'updating, but got patam_name as '
                                    f'{param_name}')

        self.eta_max = eta_max
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Validate total_steps
        if total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError('Expected positive integer total_steps, '
                                 f'but got {total_steps}')
            self.total_steps = total_steps
        else:
            self.total_steps = end - begin

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError('Expected float between 0 and 1 pct_start, '
                             f'but got {pct_start}')

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError(
                'anneal_strategy must by one of "cos" or "linear", '
                f'instead got {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

        if three_phase:
            self._schedule_phases = [
                {
                    'end_step': float(pct_start * self.total_steps) - 1,
                    f'start_{param_name}': f'initial_{param_name}',
                    f'end_{param_name}': f'max_{param_name}'
                },
                {
                    'end_step': float(2 * pct_start * self.total_steps) - 2,
                    f'start_{param_name}': f'max_{param_name}',
                    f'end_{param_name}': f'initial_{param_name}'
                },
                {
                    'end_step': self.total_steps - 1,
                    f'start_{param_name}': f'initial_{param_name}',
                    f'end_{param_name}': f'min_{param_name}'
                },
            ]
        else:
            self._schedule_phases = [
                {
                    'end_step': float(pct_start * self.total_steps) - 1,
                    f'start_{param_name}': f'initial_{param_name}',
                    f'end_{param_name}': f'max_{param_name}'
                },
                {
                    'end_step': self.total_steps - 1,
                    f'start_{param_name}': f'max_{param_name}',
                    f'end_{param_name}': f'min_{param_name}'
                },
            ]

        # Initialize parameters
        max_values = self._format_param(f'max_{param_name}', optimizer,
                                        eta_max)
        if last_step == -1:
            for idx, group in enumerate(optimizer.param_groups):
                group[f'initial_{param_name}'] = max_values[idx] / div_factor
                group[f'max_{param_name}'] = max_values[idx]
                group[f'min_{param_name}'] = \
                    group[f'initial_{param_name}'] / final_div_factor

        super().__init__(
            optimizer=optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(
                    f'expected {len(optimizer.param_groups)} values '
                    f'for {name}, got {len(param)}')
            return param
        else:
            return [param] * len(optimizer.param_groups)

    @staticmethod
    def _annealing_cos(start, end, pct):
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""

        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    def _annealing_linear(start, end, pct):
        """Linearly anneal from `start` to `end` as pct goes from 0.0 to
        1.0."""
        return (end - start) * pct + start

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              total_steps=None,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        if total_steps is not None:
            total_steps = total_steps * epoch_length
        return cls(
            *args,
            begin=begin,
            end=end,
            total_steps=total_steps,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""

        params = []
        step_num = self.last_step

        if step_num > self.total_steps:
            raise ValueError(
                f'Tried to step {step_num + 1} times. '
                f'The specified number of total steps is {self.total_steps}')

        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase['end_step']
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    computed_param = self.anneal_func(
                        group[phase['start_' + self.param_name]],
                        group[phase['end_' + self.param_name]], pct)
                    break
                start_step = phase['end_step']

            params.append(computed_param)

        return params

