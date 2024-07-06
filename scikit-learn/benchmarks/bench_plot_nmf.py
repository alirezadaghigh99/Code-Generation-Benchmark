def fit_transform(self, X, y=None, W=None, H=None):
        W, H, self.n_iter = self._fit_transform(X, W=W, H=H, update_H=True)
        self.components_ = H
        return W

def fit_transform(self, X, y=None, W=None, H=None):
        W, H, self.n_iter = self._fit_transform(X, W=W, H=H, update_H=True)
        self.components_ = H
        return W

def transform(self, X):
        check_is_fitted(self)
        H = self.components_
        W, _, self.n_iter_ = self._fit_transform(X, H=H, update_H=False)
        return W

def fit_transform(self, X, y=None, W=None, H=None):
        W, H, self.n_iter = self._fit_transform(X, W=W, H=H, update_H=True)
        self.components_ = H
        return W

