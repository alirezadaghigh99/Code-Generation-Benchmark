def logsumexp(self):
        """
        Integrates out the latent variable.
        """
        return (
            self.log_normalizer
            + torch.lgamma(self.concentration)
            - self.concentration * self.rate.log()
        )

class Gamma:
    """
    Non-normalized Gamma distribution.

        Gamma(concentration, rate) ~ (concentration - 1) * log(s) - rate * s
    """

    def __init__(self, log_normalizer, concentration, rate):
        self.log_normalizer = log_normalizer
        self.concentration = concentration
        self.rate = rate

    def log_density(self, s):
        """
        Non-normalized log probability of Gamma distribution.

        This is mainly used for testing.
        """
        return self.log_normalizer + (self.concentration - 1) * s.log() - self.rate * s

    def logsumexp(self):
        """
        Integrates out the latent variable.
        """
        return (
            self.log_normalizer
            + torch.lgamma(self.concentration)
            - self.concentration * self.rate.log()
        )

