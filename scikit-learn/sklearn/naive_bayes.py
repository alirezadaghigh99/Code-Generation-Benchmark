def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self._partial_fit(
            X, y, classes, _refit=False, sample_weight=sample_weight
        )

def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self._partial_fit(
            X, y, classes, _refit=False, sample_weight=sample_weight
        )

def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self._partial_fit(
            X, y, classes, _refit=False, sample_weight=sample_weight
        )

class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB).

    Can perform online updates to model parameters via :meth:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    Read more in the :ref:`User Guide <gaussian_naive_bayes>`.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

        .. versionadded:: 0.20

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        number of training samples observed in each class.

    class_prior_ : ndarray of shape (n_classes,)
        probability of each class.

    classes_ : ndarray of shape (n_classes,)
        class labels known to the classifier.

    epsilon_ : float
        absolute additive value to variances.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    var_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.

        .. versionadded:: 1.0

    theta_ : ndarray of shape (n_classes, n_features)
        mean of each feature per class.

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : Complement Naive Bayes classifier.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]
    """

    _parameter_constraints: dict = {
        "priors": ["array-like", None],
        "var_smoothing": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, *, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit Gaussian Naive Bayes according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17
               Gaussian Naive Bayes supports fitting with *sample_weight*.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        y = self._validate_data(y=y)
        return self._partial_fit(
            X, y, np.unique(y), _refit=True, sample_weight=sample_weight
        )

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        return self._validate_data(X, reset=False)

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        """Compute online update of Gaussian mean and variance.

        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in X is treated as independent
        -- you get variance, not covariance).

        Can take scalar mean and variance, or vector mean and variance to
        simultaneously update a number of independent Gaussians.

        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample
            weights were given, this should contain the sum of sample
            weights represented in old mean and variance.

        mu : array-like of shape (number of Gaussians,)
            Means for Gaussians in original set.

        var : array-like of shape (number of Gaussians,)
            Variances for Gaussians in original set.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        total_mu : array-like of shape (number of Gaussians,)
            Updated mean for each Gaussian over the combined set.

        total_var : array-like of shape (number of Gaussians,)
            Updated variance for each Gaussian over the combined set.
        """
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            if np.isclose(n_new, 0.0):
                return mu, var
            new_mu = np.average(X, axis=0, weights=sample_weight)
            new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        total_var = total_ssd / n_total

        return total_mu, total_var

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self._partial_fit(
            X, y, classes, _refit=False, sample_weight=sample_weight
        )

    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        """Actual implementation of Gaussian NB fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        _refit : bool, default=False
            If true, act as though this were the first time we called
            _partial_fit (ie, throw away any past fitting and start over).

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        if _refit:
            self.classes_ = None

        first_call = _check_partial_fit_first_call(self, classes)
        X, y = self._validate_data(X, y, reset=first_call)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()

        if first_call:
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.var_ = np.zeros((n_classes, n_features))

            self.class_count_ = np.zeros(n_classes, dtype=np.float64)

            # Initialise the class prior
            # Take into account the priors
            if self.priors is not None:
                priors = np.asarray(self.priors)
                # Check that the provided prior matches the number of classes
                if len(priors) != n_classes:
                    raise ValueError("Number of priors must match number of classes.")
                # Check that the sum is 1
                if not np.isclose(priors.sum(), 1.0):
                    raise ValueError("The sum of the priors should be 1.")
                # Check that the priors are non-negative
                if (priors < 0).any():
                    raise ValueError("Priors must be non-negative.")
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = np.zeros(len(self.classes_), dtype=np.float64)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.var_[:, :] -= self.epsilon_

        classes = self.classes_

        unique_y = np.unique(y)
        unique_y_in_classes = np.isin(unique_y, classes)

        if not np.all(unique_y_in_classes):
            raise ValueError(
                "The target label(s) %s in y do not exist in the initial classes %s"
                % (unique_y[~unique_y_in_classes], classes)
            )

        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.var_[i, :], X_i, sw_i
            )

            self.theta_[i, :] = new_theta
            self.var_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.var_[:, :] += self.epsilon_

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.var_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

class CategoricalNB(_BaseDiscreteNB):
    """Naive Bayes classifier for categorical features.

    The categorical Naive Bayes classifier is suitable for classification with
    discrete features that are categorically distributed. The categories of
    each feature are drawn from a categorical distribution.

    Read more in the :ref:`User Guide <categorical_naive_bayes>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 and force_alpha=True, for no smoothing).

    force_alpha : bool, default=True
        If False and alpha is less than 1e-10, it will set alpha to
        1e-10. If True, alpha will remain unchanged. This may cause
        numerical errors if alpha is too close to 0.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `force_alpha` changed to `True`.

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    min_categories : int or array-like of shape (n_features,), default=None
        Minimum number of categories per feature.

        - integer: Sets the minimum number of categories per feature to
          `n_categories` for each features.
        - array-like: shape (n_features,) where `n_categories[i]` holds the
          minimum number of categories for the ith column of the input.
        - None (default): Determines the number of categories automatically
          from the training data.

        .. versionadded:: 0.24

    Attributes
    ----------
    category_count_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the number of samples
        encountered for each class and category of the specific feature.

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_log_prob_ : list of arrays of shape (n_features,)
        Holds arrays of shape (n_classes, n_categories of respective feature)
        for each feature. Each array provides the empirical log probability
        of categories given the respective feature and class, ``P(x_i|y)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_categories_ : ndarray of shape (n_features,), dtype=np.int64
        Number of categories for each feature. This value is
        inferred from the data or set by the minimum number of categories.

        .. versionadded:: 0.24

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    ComplementNB : Complement Naive Bayes classifier.
    GaussianNB : Gaussian Naive Bayes.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> X = rng.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import CategoricalNB
    >>> clf = CategoricalNB()
    >>> clf.fit(X, y)
    CategoricalNB()
    >>> print(clf.predict(X[2:3]))
    [3]
    """

    _parameter_constraints: dict = {
        **_BaseDiscreteNB._parameter_constraints,
        "min_categories": [
            None,
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "alpha": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        alpha=1.0,
        force_alpha=True,
        fit_prior=True,
        class_prior=None,
        min_categories=None,
    ):
        super().__init__(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
        )
        self.min_categories = min_categories

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().fit(X, y, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Here, each feature of X is
            assumed to be from a different categorical distribution.
            It is further assumed that all categories of each feature are
            represented by the numbers 0, ..., n - 1, where n refers to the
            total number of categories for the given feature. This can, for
            instance, be achieved with the help of OrdinalEncoder.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().partial_fit(X, y, classes, sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        X = self._validate_data(
            X, dtype="int", accept_sparse=False, force_all_finite=True, reset=False
        )
        check_non_negative(X, "CategoricalNB (input X)")
        return X

    def _check_X_y(self, X, y, reset=True):
        X, y = self._validate_data(
            X, y, dtype="int", accept_sparse=False, force_all_finite=True, reset=reset
        )
        check_non_negative(X, "CategoricalNB (input X)")
        return X, y

    def _init_counters(self, n_classes, n_features):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.category_count_ = [np.zeros((n_classes, 0)) for _ in range(n_features)]

    @staticmethod
    def _validate_n_categories(X, min_categories):
        # rely on max for n_categories categories are encoded between 0...n-1
        n_categories_X = X.max(axis=0) + 1
        min_categories_ = np.array(min_categories)
        if min_categories is not None:
            if not np.issubdtype(min_categories_.dtype, np.signedinteger):
                raise ValueError(
                    "'min_categories' should have integral type. Got "
                    f"{min_categories_.dtype} instead."
                )
            n_categories_ = np.maximum(n_categories_X, min_categories_, dtype=np.int64)
            if n_categories_.shape != n_categories_X.shape:
                raise ValueError(
                    f"'min_categories' should have shape ({X.shape[1]},"
                    ") when an array-like is provided. Got"
                    f" {min_categories_.shape} instead."
                )
            return n_categories_
        else:
            return n_categories_X

    def _count(self, X, Y):
        def _update_cat_count_dims(cat_count, highest_feature):
            diff = highest_feature + 1 - cat_count.shape[1]
            if diff > 0:
                # we append a column full of zeros for each new category
                return np.pad(cat_count, [(0, 0), (0, diff)], "constant")
            return cat_count

        def _update_cat_count(X_feature, Y, cat_count, n_classes):
            for j in range(n_classes):
                mask = Y[:, j].astype(bool)
                if Y.dtype.type == np.int64:
                    weights = None
                else:
                    weights = Y[mask, j]
                counts = np.bincount(X_feature[mask], weights=weights)
                indices = np.nonzero(counts)[0]
                cat_count[j, indices] += counts[indices]

        self.class_count_ += Y.sum(axis=0)
        self.n_categories_ = self._validate_n_categories(X, self.min_categories)
        for i in range(self.n_features_in_):
            X_feature = X[:, i]
            self.category_count_[i] = _update_cat_count_dims(
                self.category_count_[i], self.n_categories_[i] - 1
            )
            _update_cat_count(
                X_feature, Y, self.category_count_[i], self.class_count_.shape[0]
            )

    def _update_feature_log_prob(self, alpha):
        feature_log_prob = []
        for i in range(self.n_features_in_):
            smoothed_cat_count = self.category_count_[i] + alpha
            smoothed_class_count = smoothed_cat_count.sum(axis=1)
            feature_log_prob.append(
                np.log(smoothed_cat_count) - np.log(smoothed_class_count.reshape(-1, 1))
            )
        self.feature_log_prob_ = feature_log_prob

    def _joint_log_likelihood(self, X):
        self._check_n_features(X, reset=False)
        jll = np.zeros((X.shape[0], self.class_count_.shape[0]))
        for i in range(self.n_features_in_):
            indices = X[:, i]
            jll += self.feature_log_prob_[i][:, indices].T
        total_ll = jll + self.class_log_prior_
        return total_ll

