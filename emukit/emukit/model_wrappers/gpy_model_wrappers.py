class GPyModelWrapper(
    IModel,
    IDifferentiable,
    IJointlyDifferentiable,
    ICalculateVarianceReduction,
    IEntropySearchModel,
    IPriorHyperparameters,
    IModelWithNoise,
):
    """
    This is a thin wrapper around GPy models to allow users to plug GPy models into Emukit
    """

    def __init__(self, gpy_model: GPy.core.Model, n_restarts: int = 1):
        """
        :param gpy_model: GPy model object to wrap
        :param n_restarts: Number of restarts during hyper-parameter optimization
        """
        self.model = gpy_model
        self.n_restarts = n_restarts

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        return self.model.predict(X)

    def predict_noiseless(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        return self.model.predict(X, include_likelihood=False)

    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 and n_points x n_points of the predictive
                 mean and variance at each input location
        """
        return self.model.predict(X, full_cov=True)

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get gradient of the predictions
        :return: (mean gradient, variance gradient) n_points x n_dimensions arrays of the gradients of the predictive
                 distribution at each input location
        """
        d_mean_dx, d_variance_dx = self.model.predictive_gradients(X)
        return d_mean_dx[:, :, 0], d_variance_dx

    def get_joint_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes and returns model gradients of mean and full covariance matrix at given points

        :param X: points to compute gradients at, nd array of shape (q, d)
        :return: Tuple with first item being gradient of the mean of shape (q) at X with respect to X (return shape is (q, q, d)).
                 The second item is the gradient of the full covariance matrix of shape (q, q) at X with respect to X
                 (return shape is (q, q, q, d)).
        """
        dmean_dx = dmean(X, self.model.X, self.model.kern, self.model.posterior.woodbury_vector[:, 0])
        dvariance_dx = dSigma(X, self.model.X, self.model.kern, self.model.posterior.woodbury_inv)
        return dmean_dx, dvariance_dx

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: New training features
        :param Y: New training outputs
        """
        self.model.set_XY(X, Y)

    def optimize(self, verbose=False):
        """
        Optimizes model hyper-parameters
        """
        self.model.optimize_restarts(self.n_restarts, verbose=verbose, robust=True)

    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """
        Computes the variance reduction at x_test, if a new point at x_train_new is acquired
        """
        covariance = self.model.posterior_covariance_between_points(x_train_new, x_test, include_likelihood=False)
        variance_prediction = self.model.predict(x_train_new)[1]
        return covariance**2 / variance_prediction

    def predict_covariance(self, X: np.ndarray, with_noise: bool = True) -> np.ndarray:
        """
        Calculates posterior covariance between points in X
        :param X: Array of size n_points x n_dimensions containing input locations to compute posterior covariance at
        :param with_noise: Whether to include likelihood noise in the covariance matrix
        :return: Posterior covariance matrix of size n_points x n_points
        """
        _, v = self.model.predict(X, full_cov=True, include_likelihood=with_noise)
        v = np.clip(v, 1e-10, np.inf)

        return v

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate posterior covariance between two sets of points.
        :param X1: An array of shape n_points1 x n_dimensions. This is the first argument of the
                   posterior covariance function.
        :param X2: An array of shape n_points2 x n_dimensions. This is the second argument of the
                   posterior covariance function.
        :return: An array of shape n_points1 x n_points2 of posterior covariances between X1 and X2.
            Namely, [i, j]-th entry of the returned array will represent the posterior covariance
            between i-th point in X1 and j-th point in X2.
        """
        return self.model.posterior_covariance_between_points(X1, X2, include_likelihood=False)

    @property
    def X(self) -> np.ndarray:
        """
        :return: An array of shape n_points x n_dimensions containing training inputs
        """
        return self.model.X

    @property
    def Y(self) -> np.ndarray:
        """
        :return: An array of shape n_points x 1 containing training outputs
        """
        return self.model.Y

    def generate_hyperparameters_samples(
        self, n_samples=20, n_burnin=100, subsample_interval=10, step_size=1e-1, leapfrog_steps=20
    ) -> np.ndarray:
        """
        Generates the samples from the hyper-parameters and returns them.
        :param n_samples: Number of generated samples.
        :param n_burnin: Number of initial samples not used.
        :param subsample_interval: Interval of subsampling from HMC samples.
        :param step_size: Size of the gradient steps in the HMC sampler.
        :param leapfrog_steps: Number of gradient steps before each Metropolis Hasting step.
        :return: A numpy array whose rows are samples of the hyper-parameters.

        """
        self.model.optimize(max_iters=self.n_restarts)
        # Add jitter to all unfixed parameters. After optimizing the hyperparameters, the gradient of the
        # posterior probability of the parameters wrt. the parameters will be close to 0.0, which is a poor
        # initialization for HMC
        unfixed_params = [param for param in self.model.flattened_parameters if not param.is_fixed]
        for param in unfixed_params:
            # Add jitter by multiplying with log-normal noise with mean 1 and standard deviation 0.01
            # This ensures the sign of the parameter remains the same
            param *= np.random.lognormal(np.log(1.0 / np.sqrt(1.0001)), np.sqrt(np.log(1.0001)), size=param.size)
        hmc = GPy.inference.mcmc.HMC(self.model, stepsize=step_size)
        samples = hmc.sample(num_samples=n_burnin + n_samples * subsample_interval, hmc_iters=leapfrog_steps)
        return samples[n_burnin::subsample_interval]

    def fix_model_hyperparameters(self, sample_hyperparameters: np.ndarray) -> None:
        """
        Fix model hyperparameters

        """
        if self.model._fixes_ is None:
            self.model[:] = sample_hyperparameters
        else:
            self.model[self.model._fixes_] = sample_hyperparameters
        self.model._trigger_params_changed()

