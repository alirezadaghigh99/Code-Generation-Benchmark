class GaussianHeadWithDiagonalCovariance(nn.Module):
    """Gaussian head with diagonal covariance.

    This module is intended to be attached to a neural network that outputs
    a vector that is twice the size of an action vector. The vector is split
    and interpreted as the mean and diagonal covariance of a Gaussian policy.

    Args:
        var_func (callable): Callable that computes the variance
            from the second input. It should always return positive values.
    """

    def __init__(self, var_func=nn.functional.softplus):
        super().__init__()
        self.var_func = var_func

    def forward(self, mean_and_var):
        """Return a Gaussian with given mean and diagonal covariance.

        Args:
            mean_and_var (torch.Tensor): Vector that is twice the size of an
                action vector.

        Returns:
            torch.distributions.Distribution: Gaussian distribution with given
                mean and diagonal covariance.
        """
        assert mean_and_var.ndim == 2
        mean, pre_var = mean_and_var.chunk(2, dim=1)
        scale = self.var_func(pre_var).sqrt()
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=scale), 1
        )

