class VGP(VGP_with_posterior):
    # subclassed to ensure __class__ == "VGP"

    __doc__ = VGP_deprecated.__doc__  # Use documentation from VGP_deprecated.

class VGPOpperArchambeau(GPModel, InternalDataTrainingLossMixin):
    r"""
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The key reference is :cite:t:`Opper:2009`.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior. It turns out that the optimal
    posterior precision shares off-diagonal elements with the prior, so
    only the diagonal elements of the precision need be adjusted.
    The posterior approximation is

    .. math::

       q(\mathbf f) = N(\mathbf f \,|\, \mathbf K \boldsymbol \alpha,
                         [\mathbf K^{-1} + \textrm{diag}(\boldsymbol \lambda))^2]^{-1})

    This approach has only 2ND parameters, rather than the N + N^2 of vgp,
    but the optimization is non-convex and in practice may cause difficulty.

    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
    ):
        """
        data = (X, Y) contains the input points [N, D] and the observations [N, P]
        kernel, likelihood, mean_function are appropriate GPflow objects
        """
        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.data = data_input_to_tensor(data)
        X_data, _Y_data = self.data
        self.num_data = X_data.shape[0]
        self.q_alpha = Parameter(np.zeros((self.num_data, self.num_latent_gps)))
        self.q_lambda = Parameter(
            np.ones((self.num_data, self.num_latent_gps)), transform=gpflow.utilities.positive()
        )

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        r"""
        q_alpha, q_lambda are variational parameters, size [N, R]
        This method computes the variational lower bound on the likelihood,
        which is:

        .. math::

           E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

        .. math::

           q(f) = N(f |
               K \alpha + \textrm{mean},
               [K^-1 + \textrm{diag}(\textrm{square}(\lambda))]^-1) .
        """
        X_data, Y_data = self.data

        K = self.kernel(X_data)
        K_alpha = tf.linalg.matmul(K, self.q_alpha)
        f_mean = K_alpha + self.mean_function(X_data)

        # compute the variance for each of the outputs
        I = tf.tile(
            tf.eye(self.num_data, dtype=default_float())[None, ...], [self.num_latent_gps, 1, 1]
        )
        A = (
            I
            + tf.transpose(self.q_lambda)[:, None, ...]
            * tf.transpose(self.q_lambda)[:, :, None, ...]
            * K
        )
        L = tf.linalg.cholesky(A)
        Li = tf.linalg.triangular_solve(L, I)
        tmp = Li / tf.transpose(self.q_lambda)[:, None, ...]
        f_var = 1.0 / tf.square(self.q_lambda) - tf.transpose(tf.reduce_sum(tf.square(tmp), 1))

        # some statistics about A are used in the KL
        A_logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        trAi = tf.reduce_sum(tf.square(Li))

        KL = 0.5 * (
            A_logdet
            + trAi
            - self.num_data * self.num_latent_gps
            + tf.reduce_sum(K_alpha * self.q_alpha)
        )

        v_exp = self.likelihood.variational_expectations(X_data, f_mean, f_var, Y_data)
        return tf.reduce_sum(v_exp) - KL

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        The posterior variance of F is given by

        .. math::

           q(f) = N(f |
               K \alpha + \textrm{mean}, [K^-1 + \textrm{diag}(\lambda**2)]^-1)

        Here we project this to F*, the values of the GP at Xnew which is given
        by

        .. math::

           q(F*) = N ( F* | K_{*F} \alpha + \textrm{mean}, K_{**} - K_{*f}[K_{ff} +
                                           \textrm{diag}(\lambda**-2)]^-1 K_{f*} )

        Note: This model currently does not allow full output covariances
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X_data, _ = self.data
        # compute kernel things
        Kx = self.kernel(X_data, Xnew)
        K = self.kernel(X_data)

        # predictive mean
        f_mean = tf.linalg.matmul(Kx, self.q_alpha, transpose_a=True) + self.mean_function(Xnew)

        # predictive var
        A = K + tf.linalg.diag(tf.transpose(1.0 / tf.square(self.q_lambda)))
        L = tf.linalg.cholesky(A)
        Kx_tiled = tf.tile(Kx[None, ...], [self.num_latent_gps, 1, 1])
        LiKx = tf.linalg.triangular_solve(L, Kx_tiled)
        if full_cov:
            f_var = self.kernel(Xnew) - tf.linalg.matmul(LiKx, LiKx, transpose_a=True)
        else:
            f_var = self.kernel(Xnew, full_cov=False) - tf.reduce_sum(tf.square(LiKx), axis=1)
        return f_mean, tf.transpose(f_var)

