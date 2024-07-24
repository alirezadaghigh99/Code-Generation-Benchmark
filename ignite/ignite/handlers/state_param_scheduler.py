class PiecewiseLinearStateScheduler(StateParamScheduler):
    """Piecewise linear state parameter scheduler.

    Args:
        milestones_values: list of tuples (event index, parameter value)
            represents milestones and parameter values. Milestones should be increasing integers.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        create_new: whether to create ``param_name`` on
            ``engine.state`` taking into account whether
            ``param_name`` attribute already exists or not.
            Overrides existing attribute by default, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            param_scheduler = PiecewiseLinearStateScheduler(
                param_name="param",  milestones_values=[(5, 1.0), (10, 0.8), (15, 0.6)], create_new=True
            )

            # parameter is param, milestone (5, 1.0) sets param to 1.0
            # milestone is (5, 1.0), param=1  for Epoch 1 to 5,
            # next milestone is (10, 0.8), param linearly reduces from 1.0 to 0.8
            # Epoch 10, param = 0.8
            # next milestone is (15,0.6), param linearly reduces from 0.8 to 0.6
            # Epoch 15, param = 0.6

            param_scheduler.attach(default_trainer, Events.EPOCH_COMPLETED)

            @default_trainer.on(Events.EPOCH_COMPLETED)
            def print_param():
                print(default_trainer.state.param)

            default_trainer.run([0], max_epochs=15)

        .. testoutput::

            1.0
            1.0
            1.0
            1.0
            1.0
            0.96
            0.92
            0.88
            0.8400...
            0.8
            0.76
            0.72
            0.68
            0.64
            0.6

    .. versionadded:: 0.4.7
    """

    def __init__(
        self,
        milestones_values: List[Tuple[int, float]],
        param_name: str,
        save_history: bool = False,
        create_new: bool = False,
    ):
        super(PiecewiseLinearStateScheduler, self).__init__(param_name, save_history, create_new)

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

    def get_param(self) -> Union[List[float], float]:
        start_index, end_index, start_value, end_value = self._get_start_end()
        return start_value + (end_value - start_value) * (self.event_index - start_index) / (end_index - start_index)

