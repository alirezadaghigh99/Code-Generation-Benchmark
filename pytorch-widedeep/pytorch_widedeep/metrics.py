class Accuracy(Metric):
    r"""Class to calculate the accuracy for both binary and categorical problems

    Parameters
    ----------
    top_k: int, default = 1
        Accuracy will be computed using the top k most likely classes in
        multiclass problems

    Examples
    --------
    >>> import torch
    >>>
    >>> from pytorch_widedeep.metrics import Accuracy
    >>>
    >>> acc = Accuracy()
    >>> y_true = torch.tensor([0, 1, 0, 1]).view(-1, 1)
    >>> y_pred = torch.tensor([[0.3, 0.2, 0.6, 0.7]]).view(-1, 1)
    >>> acc(y_pred, y_true)
    array(0.5)
    >>>
    >>> acc = Accuracy(top_k=2)
    >>> y_true = torch.tensor([0, 1, 2])
    >>> y_pred = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.1, 0.8], [0.1, 0.5, 0.4]])
    >>> acc(y_pred, y_true)
    array(0.66666667)
    """

    def __init__(self, top_k: int = 1):
        super(Accuracy, self).__init__()

        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0
        self._name = "acc"

    def reset(self):
        """
        resets counters to 0
        """
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        num_classes = y_pred.size(1)

        if num_classes == 1:
            y_pred = y_pred.round()
            y_true = y_true
        elif num_classes > 1:
            y_pred = y_pred.topk(self.top_k, 1)[1]
            y_true = y_true.view(-1, 1).expand_as(y_pred)

        self.correct_count += y_pred.eq(y_true).sum().item()  # type: ignore[assignment]
        self.total_count += len(y_pred)
        accuracy = float(self.correct_count) / float(self.total_count)
        return np.array(accuracy)

class R2Score(Metric):
    r"""
    Calculates R-Squared, the
    [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination>):

    $$
    R^2 = 1 - \frac{\sum_{j=1}^n(y_j - \hat{y_j})^2}{\sum_{j=1}^n(y_j - \bar{y})^2}
    $$

    where $\hat{y_j}$ is the ground truth, $y_j$ is the predicted value and
    $\bar{y}$ is the mean of the ground truth.

    Examples
    --------
    >>> import torch
    >>>
    >>> from pytorch_widedeep.metrics import R2Score
    >>>
    >>> r2 = R2Score()
    >>> y_true = torch.tensor([3, -0.5, 2, 7]).view(-1, 1)
    >>> y_pred = torch.tensor([2.5, 0.0, 2, 8]).view(-1, 1)
    >>> r2(y_pred, y_true)
    array(0.94860814)
    """

    def __init__(self):
        self.numerator = 0
        self.denominator = 0
        self.num_examples = 0
        self.y_true_sum = 0

        self._name = "r2"

    def reset(self):
        """
        resets counters to 0
        """
        self.numerator = 0
        self.denominator = 0
        self.num_examples = 0
        self.y_true_sum = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        self.numerator += ((y_pred - y_true) ** 2).sum().item()

        self.num_examples += y_true.shape[0]
        self.y_true_sum += y_true.sum().item()
        y_true_avg = self.y_true_sum / self.num_examples
        self.denominator += ((y_true - y_true_avg) ** 2).sum().item()
        return np.array((1 - (self.numerator / self.denominator)))

