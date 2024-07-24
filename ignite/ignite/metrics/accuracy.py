class Accuracy(_BaseClassification):
    r"""Calculates the accuracy for binary, multiclass and multilabel data.

    .. math:: \text{Accuracy} = \frac{ TP + TN }{ TP + TN + FP + FN }

    where :math:`\text{TP}` is true positives, :math:`\text{TN}` is true negatives,
    :math:`\text{FP}` is false positives and :math:`\text{FN}` is false negatives.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).
    - `y` and `y_pred` must be in the following shape of (batch_size, num_categories, ...) and
      num_categories must be greater than 1 for multilabel cases.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        is_multilabel: flag to use in multilabel case. By default, False.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        Binary case

        .. testcode:: 1

            metric = Accuracy()
            metric.attach(default_evaluator, "accuracy")
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["accuracy"])

        .. testoutput:: 1

            0.6666...

        Multiclass case

        .. testcode:: 2

            metric = Accuracy()
            metric.attach(default_evaluator, "accuracy")
            y_true = torch.tensor([2, 0, 2, 1, 0, 1])
            y_pred = torch.tensor([
                [0.0266, 0.1719, 0.3055],
                [0.6886, 0.3978, 0.8176],
                [0.9230, 0.0197, 0.8395],
                [0.1785, 0.2670, 0.6084],
                [0.8448, 0.7177, 0.7288],
                [0.7748, 0.9542, 0.8573],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["accuracy"])

        .. testoutput:: 2

            0.5

        Multilabel case

        .. testcode:: 3

            metric = Accuracy(is_multilabel=True)
            metric.attach(default_evaluator, "accuracy")
            y_true = torch.tensor([
                [0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 0, 1],
            ])
            y_pred = torch.tensor([
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 0, 1],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["accuracy"])

        .. testoutput:: 3

            0.2

        In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of
        predictions can be done as below:

        .. testcode:: 4

            def thresholded_output_transform(output):
                y_pred, y = output
                y_pred = torch.round(y_pred)
                return y_pred, y

            metric = Accuracy(output_transform=thresholded_output_transform)
            metric.attach(default_evaluator, "accuracy")
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([0.6, 0.2, 0.9, 0.4, 0.7, 0.65])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["accuracy"])

        .. testoutput:: 4

            0.6666...
    """

    _state_dict_all_req_keys = ("_num_correct", "_num_examples")

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Accuracy, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(Accuracy, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if self._type == "binary":
            correct = torch.eq(y_pred.view(-1).to(y), y.view(-1))
        elif self._type == "multiclass":
            indices = torch.argmax(y_pred, dim=1)
            correct = torch.eq(indices, y).view(-1)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (N x ..., C)
            num_classes = y_pred.size(1)
            last_dim = y_pred.ndimension()
            y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
            y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
            correct = torch.all(y == y_pred.type_as(y), dim=-1)

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")
        return self._num_correct.item() / self._num_examples

