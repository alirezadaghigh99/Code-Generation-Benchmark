def logsumexp(self):
        """
        Integrates out the latent variable.
        """
        return (
            self.log_normalizer
            + torch.lgamma(self.concentration)
            - self.concentration * self.rate.log()
        )

