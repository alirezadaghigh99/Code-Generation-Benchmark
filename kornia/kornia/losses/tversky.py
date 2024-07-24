class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Args:
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.

    Shape:
        - Pred: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C-1`.

    Examples:
        >>> N = 5  # num_classes
        >>> criterion = TverskyLoss(alpha=0.5, beta=0.5)
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, beta: float, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tversky_loss(pred, target, self.alpha, self.beta, self.eps)

