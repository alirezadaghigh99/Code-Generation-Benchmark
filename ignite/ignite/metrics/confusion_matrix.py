class ConfusionMatrix(Metric):
    """Calculates confusion matrix for multi-class data.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - `y_pred` must contain logits and has the following shape (batch_size, num_classes, ...).
      If you are doing binary classification, see Note for an example on how to get this.
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
      with or without the background class. During the computation, argmax of `y_pred` is taken to determine
      predicted classes.

    Args:
        num_classes: Number of classes, should be > 1. See notes for more details.
        average: confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Note:
        The confusion matrix is formatted such that columns are predictions and rows are targets.
        For example, if you were to plot the matrix, you could correctly assign to the horizontal axis
        the label "predicted values" and to the vertical axis the label "actual values".

    Note:
        In case of the targets `y` in `(batch_size, ...)` format, target indices between 0 and `num_classes` only
        contribute to the confusion matrix and others are neglected. For example, if `num_classes=20` and target index
        equal 255 is encountered, then it is filtered out.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode:: 1

            metric = ConfusionMatrix(num_classes=3)
            metric.attach(default_evaluator, 'cm')
            y_true = torch.tensor([0, 1, 0, 1, 2])
            y_pred = torch.tensor([
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['cm'])

        .. testoutput:: 1

            tensor([[1, 1, 0],
                    [0, 2, 0],
                    [0, 1, 0]])

        If you are doing binary classification with a single output unit, you may have to transform your network output,
        so that you have one value for each class. E.g. you can transform your network output into a one-hot vector
        with:

        .. testcode:: 2

            def binary_one_hot_output_transform(output):
                from ignite import utils
                y_pred, y = output
                y_pred = torch.sigmoid(y_pred).round().long()
                y_pred = utils.to_onehot(y_pred, 2)
                y = y.long()
                return y_pred, y

            metric = ConfusionMatrix(num_classes=2, output_transform=binary_one_hot_output_transform)
            metric.attach(default_evaluator, 'cm')
            y_true = torch.tensor([0, 1, 0, 1, 0])
            y_pred = torch.tensor([0, 0, 1, 1, 0])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['cm'])

        .. testoutput:: 2

            tensor([[2, 1],
                    [1, 1]])
    """

    _state_dict_all_req_keys = ("confusion_matrix", "_num_examples")

    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of 'samples', 'recall', 'precision'")

        if num_classes <= 1:
            raise ValueError("Argument num_classes needs to be > 1")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        super(ConfusionMatrix, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64, device=self._device)
        self._num_examples = 0

    def _check_shape(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() < 2:
            raise ValueError(
                f"y_pred must have shape (batch_size, num_classes (currently set to {self.num_classes}), ...), "
                f"but given {y_pred.shape}"
            )

        if y_pred.shape[1] != self.num_classes:
            raise ValueError(f"y_pred does not have correct number of classes: {y_pred.shape[1]} vs {self.num_classes}")

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                f"y_pred must have shape (batch_size, num_classes (currently set to {self.num_classes}), ...) "
                "and y must have shape of (batch_size, ...), "
                f"but given {y.shape} vs {y_pred.shape}."
            )

        y_shape = y.shape
        y_pred_shape: Tuple[int, ...] = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output[0].detach(), output[1].detach()

        self._num_examples += y_pred.shape[0]

        # target is (batch_size, ...)
        y_pred = torch.argmax(y_pred, dim=1).flatten()
        y = y.flatten()

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    @sync_all_reduce("confusion_matrix", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Confusion matrix must have at least one example before it can be computed.")
        if self.average:
            self.confusion_matrix = self.confusion_matrix.float()
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            else:
                return self.normalize(self.confusion_matrix, self.average)
        return self.confusion_matrix

    @staticmethod
    def normalize(matrix: torch.Tensor, average: str) -> torch.Tensor:
        """Normalize given `matrix` with given `average`."""
        if average == "recall":
            return matrix / (matrix.sum(dim=1).unsqueeze(1) + 1e-15)
        elif average == "precision":
            return matrix / (matrix.sum(dim=0) + 1e-15)
        else:
            raise ValueError("Argument average should be one of 'samples', 'recall', 'precision'")

