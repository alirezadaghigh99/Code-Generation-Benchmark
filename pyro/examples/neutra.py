    def sample(self, sample_shape=()):
        u = self.mvn.sample(sample_shape)
        u0, u1 = u[..., 0], u[..., 1]
        a, b = self.a, self.b
        x = a * u0
        y = (u1 / a) + b * (u0**2 + a**2)
        return torch.stack([x, y], -1)