def predict(self, x: np.array) -> np.array:
        """Returns the prediction made by the underlying model.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of test examples.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted outputs of test examples.
        """
        if (
                not self.clean or
                self.last_x is None or
                self.last_y is None or
                not np.array_equal(self.last_x, x)
        ):
            self.last_x = x
            self.last_y = self._underlying_predict(x)
            self.clean = True

        return self.last_y.copy()

