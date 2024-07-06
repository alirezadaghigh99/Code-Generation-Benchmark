def fit_transform(self, X, y=None, W=None, H=None):
        W, H, self.n_iter = self._fit_transform(X, W=W, H=H, update_H=True)
        self.components_ = H
        return W

