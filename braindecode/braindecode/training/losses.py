class TimeSeriesLoss(nn.Module):
    """Compute Loss between timeseries targets and predictions.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)
    Assumes targets are in shape:
    n_batch size x n_classes x window_len (in time)
    If the targets contain NaNs, the NaNs will be masked out and the loss will be only computed for
    predictions valid corresponding to valid target values."""

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, preds, targets):
        """Forward pass.

        Parameters
        ----------
        preds: torch.Tensor
            Model's prediction with shape (batch_size, n_classes, n_times).
        targets: torch.Tensor
            Target labels with shape (batch_size, n_classes, n_times).
        """
        n_preds = preds.shape[-1]
        # slice the targets to fit preds shape
        targets = targets[:, :, -n_preds:]
        # create valid targets mask
        mask = ~torch.isnan(targets)
        # select valid targets that have a matching predictions
        masked_targets = targets[mask]
        masked_preds = preds[mask]
        return self.loss_function(masked_preds, masked_targets)

