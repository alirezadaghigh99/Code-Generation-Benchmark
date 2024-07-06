def sample(self, mean, noise_level):
            return max(self._random_state.normal(mean, mean * noise_level), 5e-9)

