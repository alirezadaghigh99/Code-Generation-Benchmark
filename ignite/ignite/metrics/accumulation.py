class VariableAccumulation(Metric):
    """Single variable accumulator helper to compute (arithmetic, geometric, harmonic) average of a single variable.

    - ``update`` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.

    Note:

        The class stores input into two public variables: `accumulator` and `num_examples`.
        Number of samples is updated following the rule:

        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is a ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).

    Args:
        op: a callable to update accumulator. Method's signature is `(accumulator, output)`.
            For example, to compute arithmetic mean value, `op = lambda a, x: a + x`.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    """

    required_output_keys = None
    _state_dict_all_req_keys = ("accumulator", "num_examples")

    def __init__(
        self,
        op: Callable,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if not callable(op):
            raise TypeError(f"Argument op should be a callable, but given {type(op)}")

        self._op = op

        super(VariableAccumulation, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.accumulator = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self.num_examples = 0

    def _check_output_type(self, output: Union[float, torch.Tensor]) -> None:
        if not isinstance(output, (numbers.Number, torch.Tensor)):
            raise TypeError(f"Output should be a number or torch.Tensor, but given {type(output)}")

    @reinit__is_reduced
    def update(self, output: Union[float, torch.Tensor]) -> None:
        self._check_output_type(output)

        if isinstance(output, torch.Tensor):
            output = output.detach()
            if not (output.device == self._device and output.dtype == self.accumulator.dtype):
                output = output.to(self.accumulator)

        self.accumulator = self._op(self.accumulator, output)

        if isinstance(output, torch.Tensor):
            self.num_examples += output.shape[0] if len(output.shape) > 1 else 1
        else:
            self.num_examples += 1

    @sync_all_reduce("accumulator", "num_examples")
    def compute(self) -> Tuple[torch.Tensor, int]:
        return self.accumulator, self.num_examples

