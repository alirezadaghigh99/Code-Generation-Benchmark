class AggregationMetric(Metric):
    """
    Calculate metric on mean prediction and actuals.
    """

    def __init__(self, metric: Metric, **kwargs):
        """
        Args:
            metric (Metric): metric which to calculate on aggreation.
        """
        super().__init__(**kwargs)
        self.metric = metric

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        y_pred_mean, y_mean = self._calculate_mean(y_pred, y_actual)
        # update metric. unsqueeze first batch dimension (as batches are collapsed)
        self.metric.update(y_pred_mean, y_mean, **kwargs)

    @staticmethod
    def _calculate_mean(y_pred: torch.Tensor, y_actual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract target and weight
        if isinstance(y_actual, (tuple, list)) and not isinstance(y_actual, rnn.PackedSequence):
            target, weight = y_actual
        else:
            target = y_actual
            weight = None

        # handle rnn sequence as target
        if isinstance(target, rnn.PackedSequence):
            target, lengths = rnn.pad_packed_sequence(target, batch_first=True)
            # batch sizes reside on the CPU by default -> we need to bring them to GPU
            lengths = lengths.to(target.device)

            # calculate mask for time steps
            length_mask = create_mask(target.size(1), lengths, inverse=True)

            # modify weight
            if weight is None:
                weight = length_mask
            else:
                weight = weight * length_mask

        if weight is None:
            y_mean = target.mean(0)
            y_pred_mean = y_pred.mean(0)
        else:
            # calculate weighted sums
            y_mean = (target * unsqueeze_like(weight, y_pred)).sum(0) / weight.sum(0)

            y_pred_sum = (y_pred * unsqueeze_like(weight, y_pred)).sum(0)
            y_pred_mean = y_pred_sum / unsqueeze_like(weight.sum(0), y_pred_sum)
        return y_pred_mean.unsqueeze(0), y_mean.unsqueeze(0)

    def compute(self):
        return self.metric.compute()

    @torch.jit.unused
    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs):
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values
            **kwargs: arguments to update function

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        y_pred_mean, y_mean = self._calculate_mean(y_pred, y_actual)
        return self.metric(y_pred_mean, y_mean, **kwargs)

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute

    def _sync_dist(self, dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None) -> None:
        # No syncing required here. syncing will be done in metrics
        pass

    def reset(self) -> None:
        self.metrics.reset()

    def persistent(self, mode: bool = False) -> None:
        self.metric.persistent(mode=mode)

