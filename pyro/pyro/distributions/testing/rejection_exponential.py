class RejectionExponential(Rejector):
    arg_constraints = {"rate": constraints.positive, "factor": constraints.positive}
    support = constraints.positive

    def __init__(self, rate, factor):
        assert (factor <= 1).all()
        self.rate, self.factor = broadcast_all(rate, factor)
        propose = Exponential(self.factor * self.rate)
        log_scale = self.factor.log()
        super().__init__(propose, self.log_prob_accept, log_scale)

    @weakmethod
    def log_prob_accept(self, x):
        result = (self.factor - 1) * self.rate * x
        assert result.max() <= 0, result.max()
        return result

    @property
    def batch_shape(self):
        return self.rate.shape

    @property
    def event_shape(self):
        return torch.Size()

