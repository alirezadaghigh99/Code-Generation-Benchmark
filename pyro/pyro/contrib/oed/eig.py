class EwmaLog:
    """Logarithm function with exponentially weighted moving average
    for gradients.

    For input `inputs` this function return :code:`inputs.log()`. However, it
    computes the gradient as

        :math:`\\frac{\\sum_{t=0}^{T-1} \\alpha^t}{\\sum_{t=0}^{T-1} \\alpha^t x_{T-t}}`

    where :math:`x_t` are historical input values passed to this function,
    :math:`x_T` being the most recently seen value.

    This gradient may help with numerical stability when the sequence of
    inputs to the function form a convergent sequence.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.ewma = 0.0
        self.n = 0
        self.s = 0.0

    def __call__(self, inputs, s, dim=0, keepdim=False):
        """Updates the moving average, and returns :code:`inputs.log()`."""
        self.n += 1
        if torch_isnan(self.ewma) or torch_isinf(self.ewma):
            ewma = inputs
        else:
            ewma = inputs * (1.0 - self.alpha) / (1 - self.alpha**self.n) + torch.exp(
                self.s - s
            ) * self.ewma * (self.alpha - self.alpha**self.n) / (
                1 - self.alpha**self.n
            )
        self.ewma = ewma.detach()
        self.s = s.detach()
        return _ewma_log_fn(inputs, ewma)

