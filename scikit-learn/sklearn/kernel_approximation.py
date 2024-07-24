class AdditiveChi2Sampler(TransformerMixin, BaseEstimator):
    """Approximate feature map for additive chi2 kernel.

    Uses sampling the fourier transform of the kernel characteristic
    at regular intervals.

    Since the kernel that is to be approximated is additive, the components of
    the input vectors can be treated separately.  Each entry in the original
    space is transformed into 2*sample_steps-1 features, where sample_steps is
    a parameter of the method. Typical values of sample_steps include 1, 2 and
    3.

    Optimal choices for the sampling interval for certain data ranges can be
    computed (see the reference). The default values should be reasonable.

    Read more in the :ref:`User Guide <additive_chi_kernel_approx>`.

    Parameters
    ----------
    sample_steps : int, default=2
        Gives the number of (complex) sampling points.

    sample_interval : float, default=None
        Sampling interval. Must be specified when sample_steps not in {1,2,3}.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SkewedChi2Sampler : A Fourier-approximation to a non-additive variant of
        the chi squared kernel.

    sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.

    sklearn.metrics.pairwise.additive_chi2_kernel : The exact additive chi
        squared kernel.

    Notes
    -----
    This estimator approximates a slightly different version of the additive
    chi squared kernel then ``metric.additive_chi2`` computes.

    This estimator is stateless and does not need to be fitted. However, we
    recommend to call :meth:`fit_transform` instead of :meth:`transform`, as
    parameter validation is only performed in :meth:`fit`.

    References
    ----------
    See `"Efficient additive kernels via explicit feature maps"
    <http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf>`_
    A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence,
    2011

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.kernel_approximation import AdditiveChi2Sampler
    >>> X, y = load_digits(return_X_y=True)
    >>> chi2sampler = AdditiveChi2Sampler(sample_steps=2)
    >>> X_transformed = chi2sampler.fit_transform(X, y)
    >>> clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)
    >>> clf.fit(X_transformed, y)
    SGDClassifier(max_iter=5, random_state=0)
    >>> clf.score(X_transformed, y)
    0.9499...
    """

    _parameter_constraints: dict = {
        "sample_steps": [Interval(Integral, 1, None, closed="left")],
        "sample_interval": [Interval(Real, 0, None, closed="left"), None],
    }

    def __init__(self, *, sample_steps=2, sample_interval=None):
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = self._validate_data(X, accept_sparse="csr")
        check_non_negative(X, "X in AdditiveChi2Sampler.fit")

        if self.sample_interval is None and self.sample_steps not in (1, 2, 3):
            raise ValueError(
                "If sample_steps is not in [1, 2, 3],"
                " you need to provide sample_interval"
            )

        return self

    def transform(self, X):
        """Apply approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, \
               shape = (n_samples, n_features * (2*sample_steps - 1))
            Whether the return value is an array or sparse matrix depends on
            the type of the input X.
        """
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        check_non_negative(X, "X in AdditiveChi2Sampler.transform")
        sparse = sp.issparse(X)

        if self.sample_interval is None:
            # See figure 2 c) of "Efficient additive kernels via explicit feature maps" # noqa
            # <http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf>
            # A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence, # noqa
            # 2011
            if self.sample_steps == 1:
                sample_interval = 0.8
            elif self.sample_steps == 2:
                sample_interval = 0.5
            elif self.sample_steps == 3:
                sample_interval = 0.4
            else:
                raise ValueError(
                    "If sample_steps is not in [1, 2, 3],"
                    " you need to provide sample_interval"
                )
        else:
            sample_interval = self.sample_interval

        # zeroth component
        # 1/cosh = sech
        # cosh(0) = 1.0
        transf = self._transform_sparse if sparse else self._transform_dense
        return transf(X, self.sample_steps, sample_interval)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Only used to validate feature names with the names seen in :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(
            self, input_features, generate_names=True
        )
        est_name = self.__class__.__name__.lower()

        names_list = [f"{est_name}_{name}_sqrt" for name in input_features]

        for j in range(1, self.sample_steps):
            cos_names = [f"{est_name}_{name}_cos{j}" for name in input_features]
            sin_names = [f"{est_name}_{name}_sin{j}" for name in input_features]
            names_list.extend(cos_names + sin_names)

        return np.asarray(names_list, dtype=object)

    @staticmethod
    def _transform_dense(X, sample_steps, sample_interval):
        non_zero = X != 0.0
        X_nz = X[non_zero]

        X_step = np.zeros_like(X)
        X_step[non_zero] = np.sqrt(X_nz * sample_interval)

        X_new = [X_step]

        log_step_nz = sample_interval * np.log(X_nz)
        step_nz = 2 * X_nz * sample_interval

        for j in range(1, sample_steps):
            factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * sample_interval))

            X_step = np.zeros_like(X)
            X_step[non_zero] = factor_nz * np.cos(j * log_step_nz)
            X_new.append(X_step)

            X_step = np.zeros_like(X)
            X_step[non_zero] = factor_nz * np.sin(j * log_step_nz)
            X_new.append(X_step)

        return np.hstack(X_new)

    @staticmethod
    def _transform_sparse(X, sample_steps, sample_interval):
        indices = X.indices.copy()
        indptr = X.indptr.copy()

        data_step = np.sqrt(X.data * sample_interval)
        X_step = sp.csr_matrix(
            (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
        )
        X_new = [X_step]

        log_step_nz = sample_interval * np.log(X.data)
        step_nz = 2 * X.data * sample_interval

        for j in range(1, sample_steps):
            factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * sample_interval))

            data_step = factor_nz * np.cos(j * log_step_nz)
            X_step = sp.csr_matrix(
                (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
            )
            X_new.append(X_step)

            data_step = factor_nz * np.sin(j * log_step_nz)
            X_step = sp.csr_matrix(
                (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
            )
            X_new.append(X_step)

        return sp.hstack(X_new)

    def _more_tags(self):
        return {"stateless": True, "requires_positive_X": True}

class RBFSampler(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Approximate a RBF kernel feature map using random Fourier features.

    It implements a variant of Random Kitchen Sinks.[1]

    Read more in the :ref:`User Guide <rbf_kernel_approx>`.

    Parameters
    ----------
    gamma : 'scale' or float, default=1.0
        Parameter of RBF kernel: exp(-gamma * x^2).
        If ``gamma='scale'`` is passed then it uses
        1 / (n_features * X.var()) as value of gamma.

        .. versionadded:: 1.2
           The option `"scale"` was added in 1.2.

    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    random_offset_ : ndarray of shape (n_components,), dtype={np.float64, np.float32}
        Random offset used to compute the projection in the `n_components`
        dimensions of the feature space.

    random_weights_ : ndarray of shape (n_features, n_components),\
        dtype={np.float64, np.float32}
        Random projection directions drawn from the Fourier transform
        of the RBF kernel.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    Nystroem : Approximate a kernel map using a subset of the training data.
    PolynomialCountSketch : Polynomial kernel approximation via Tensor Sketch.
    SkewedChi2Sampler : Approximate feature map for
        "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.

    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)

    Examples
    --------
    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSampler(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=5)
    >>> clf.score(X_features, y)
    1.0
    """

    _parameter_constraints: dict = {
        "gamma": [
            StrOptions({"scale"}),
            Interval(Real, 0.0, None, closed="left"),
        ],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(self, *, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X.

        Samples random projection according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, accept_sparse="csr")
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]
        sparse = sp.issparse(X)
        if self.gamma == "scale":
            # var = E[X^2] - E[X]^2 if sparse
            X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
            self._gamma = 1.0 / (n_features * X_var) if X_var != 0 else 1.0
        else:
            self._gamma = self.gamma
        self.random_weights_ = (2.0 * self._gamma) ** 0.5 * random_state.normal(
            size=(n_features, self.n_components)
        )

        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)

        if X.dtype == np.float32:
            # Setting the data type of the fitted attribute will ensure the
            # output data type during `transform`.
            self.random_weights_ = self.random_weights_.astype(X.dtype, copy=False)
            self.random_offset_ = self.random_offset_.astype(X.dtype, copy=False)

        self._n_features_out = self.n_components
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= (2.0 / self.n_components) ** 0.5
        return projection

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}

class SkewedChi2Sampler(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Approximate feature map for "skewed chi-squared" kernel.

    Read more in the :ref:`User Guide <skewed_chi_kernel_approx>`.

    Parameters
    ----------
    skewedness : float, default=1.0
        "skewedness" parameter of the kernel. Needs to be cross-validated.

    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    random_weights_ : ndarray of shape (n_features, n_components)
        Weight array, sampled from a secant hyperbolic distribution, which will
        be used to linearly transform the log of the data.

    random_offset_ : ndarray of shape (n_features, n_components)
        Bias term, which will be added to the data. It is uniformly distributed
        between 0 and 2*pi.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    Nystroem : Approximate a kernel map using a subset of the training data.
    RBFSampler : Approximate a RBF kernel feature map using random Fourier
        features.
    SkewedChi2Sampler : Approximate feature map for "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    References
    ----------
    See "Random Fourier Approximations for Skewed Multiplicative Histogram
    Kernels" by Fuxin Li, Catalin Ionescu and Cristian Sminchisescu.

    Examples
    --------
    >>> from sklearn.kernel_approximation import SkewedChi2Sampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> chi2_feature = SkewedChi2Sampler(skewedness=.01,
    ...                                  n_components=10,
    ...                                  random_state=0)
    >>> X_features = chi2_feature.fit_transform(X, y)
    >>> clf = SGDClassifier(max_iter=10, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=10)
    >>> clf.score(X_features, y)
    1.0
    """

    _parameter_constraints: dict = {
        "skewedness": [Interval(Real, None, None, closed="neither")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(self, *, skewedness=1.0, n_components=100, random_state=None):
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X.

        Samples random projection according to n_features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X)
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]
        uniform = random_state.uniform(size=(n_features, self.n_components))
        # transform by inverse CDF of sech
        self.random_weights_ = 1.0 / np.pi * np.log(np.tan(np.pi / 2.0 * uniform))
        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)

        if X.dtype == np.float32:
            # Setting the data type of the fitted attribute will ensure the
            # output data type during `transform`.
            self.random_weights_ = self.random_weights_.astype(X.dtype, copy=False)
            self.random_offset_ = self.random_offset_.astype(X.dtype, copy=False)

        self._n_features_out = self.n_components
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features. All values of X must be
            strictly greater than "-skewedness".

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, copy=True, dtype=[np.float64, np.float32], reset=False
        )
        if (X <= -self.skewedness).any():
            raise ValueError("X may not contain entries smaller than -skewedness.")

        X += self.skewedness
        np.log(X, X)
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= np.sqrt(2.0) / np.sqrt(self.n_components)
        return projection

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}

