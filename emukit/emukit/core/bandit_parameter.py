def round(self, x: np.ndarray) -> np.ndarray:
        """
        Rounds each row in x to represent a valid value for this bandit variable. Note that this
        valid value may be 'far' from the suggested value.

        :param x: A 2d array NxD to be rounded (D is len(self.parameters))
        :returns: An array NxD where each row represents a value from the domain
                  that is closest to the corresponding row in x
        """
        if x.ndim != 2:
            raise ValueError("Expected 2d array, got {}".format(x.ndim))

        if x.shape[1] != self.dimension:
            raise ValueError("Expected {} column array, got {}".format(self.dimension, x.shape[1]))

        x_rounded = []
        for row in x:
            dists = np.sqrt(np.sum((self.domain - row) ** 2))
            rounded_value = min(self.domain, key=lambda d: np.linalg.norm(d - row))
            x_rounded.append(rounded_value)

        if not all([self.check_in_domain(xr) for xr in x_rounded]):
            raise ValueError("Rounding error encountered, not all rounded values in domain.")
        return np.row_stack(x_rounded)

