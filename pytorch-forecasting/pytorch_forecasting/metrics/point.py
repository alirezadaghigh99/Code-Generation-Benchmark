class MAE(MultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss

class TweedieLoss(MultiHorizonMetric):
    """
    Tweedie loss.

    Tweedie regression with log-link. It might be useful, e.g., for modeling total
    loss in insurance, or for any target that might be tweedie-distributed.

    The loss will take the exponential of the network output before it is returned as prediction.
    Target normalizer should therefore have no "reverse" transformation, e.g.
    for the :py:class:`~data.timeseries.TimeSeriesDataSet` initialization, one could use:

    .. code-block:: python

        from pytorch_forecasting import TimeSeriesDataSet, EncoderNormalizer

        dataset = TimeSeriesDataSet(
            target_normalizer=EncoderNormalizer(transformation=dict(forward=torch.log1p))
        )

    Note that in this example, the data is log1p-transformed before normalized but not re-transformed.
    The TweedieLoss applies this "exp"-re-transformation on the network output after it has been de-normalized.
    The result is the model prediction.
    """

    def __init__(self, reduction="mean", p: float = 1.5, **kwargs):
        """
        Args:
            p (float, optional): tweedie variance power which is greater equal
                1.0 and smaller 2.0. Close to ``2`` shifts to
                Gamma distribution and close to ``1`` shifts to Poisson distribution.
                Defaults to 1.5.
            reduction (str, optional): How to reduce the loss. Defaults to "mean".
        """
        super().__init__(reduction=reduction, **kwargs)
        assert 1 <= p < 2, "p must be in range [1, 2]"
        self.p = p

    def to_prediction(self, out: Dict[str, torch.Tensor]):
        rate = torch.exp(super().to_prediction(out))
        return rate

    def loss(self, y_pred, y_true):
        y_pred = super().to_prediction(y_pred)
        a = y_true * torch.exp(y_pred * (1 - self.p)) / (1 - self.p)
        b = torch.exp(y_pred * (2 - self.p)) / (2 - self.p)
        loss = -a + b
        return loss

class SMAPE(MultiHorizonMetric):
    """
    Symmetric mean absolute percentage. Assumes ``y >= 0``.

    Defined as ``2*(y - y_pred).abs() / (y.abs() + y_pred.abs())``
    """

    def loss(self, y_pred, target):
        y_pred = self.to_prediction(y_pred)
        loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
        return loss

