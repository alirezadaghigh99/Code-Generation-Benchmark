class SGPR(SGPR_with_posterior):
    # subclassed to ensure __class__ == "SGPR"

    __doc__ = SGPR_deprecated.__doc__  # Use documentation from SGPR_deprecated.

class GPRFITC(SGPRBase_deprecated):
    """
    This implements GP regression with the FITC approximation.

    The key reference is :cite:t:`Snelson06sparsegaussian`.

    Implementation loosely based on code from GPML matlab library although
    obviously gradients are automatic in GPflow.
    """

    @check_shapes(
        "return[0]: [N, R]",
        "return[1]: [N]",
        "return[2]: [M, M]",
        "return[3]: [M, M]",
        "return[4]: [M, R]",
        "return[5]: [N, R]",
        "return[6]: [M, R]",
    )
    def common_terms(
        self,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)  # size [N, R]
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        sigma_sq = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)

        Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
        V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + sigma_sq

        B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(
            V / nu, V, transpose_b=True
        )
        L = tf.linalg.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size [N, R]
        alpha = tf.linalg.matmul(V, beta)  # size [M, R]

        gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [M, R]

        return err, nu, Luu, L, alpha, beta, gamma

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.fitc_log_marginal_likelihood()

    @check_shapes(
        "return: []",
    )
    def fitc_log_marginal_likelihood(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        # FITC approximation to the log marginal likelihood is
        # log ( normal( y | mean, K_fitc ) )
        # where K_fitc = Qff + diag( \nu )
        # where Qff = Kfu kuu^{-1} kuf
        # with \nu_i = Kff_{i,i} - Qff_{i,i} + \sigma^2

        # We need to compute the Mahalanobis term -0.5* err^T K_fitc^{-1} err
        # (summed over functions).

        # We need to deal with the matrix inverse term.
        # K_fitc^{-1} = ( Qff + \diag( \nu ) )^{-1}
        #             = ( V^T V + \diag( \nu ) )^{-1}
        # Applying the Woodbury identity we obtain
        #             = \diag( \nu^{-1} )
        #                 - \diag( \nu^{-1} ) V^T ( I + V \diag( \nu^{-1} ) V^T )^{-1}
        #                     V \diag(\nu^{-1} )
        # Let \beta =  \diag( \nu^{-1} ) err
        # and let \alpha = V \beta
        # then Mahalanobis term = -0.5* (
        #    \beta^T err - \alpha^T Solve( I + V \diag( \nu^{-1} ) V^T, alpha )
        # )

        err, nu, _Luu, L, _alpha, _beta, gamma = self.common_terms()

        mahalanobisTerm = -0.5 * tf.reduce_sum(
            tf.square(err) / tf.expand_dims(nu, 1)
        ) + 0.5 * tf.reduce_sum(tf.square(gamma))

        # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

        # We need to deal with the log determinant term.
        # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
        #                     = \log \det( V^T V + \diag( \nu ) )
        # Applying the determinant lemma we obtain
        #                     = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
        #                     = \log [
        #                        \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T )
        #                     ]

        constantTerm = -0.5 * self.num_data * tf.math.log(tf.constant(2.0 * np.pi, default_float()))
        logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(nu)) - tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L))
        )
        logNormalizingTerm = constantTerm + logDeterminantTerm

        return mahalanobisTerm + logNormalizingTerm * self.num_latent_gps

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        _, _, Luu, L, _, _, gamma = self.common_terms()
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, N]

        w = tf.linalg.triangular_solve(Luu, Kus, lower=True)  # [M, N]

        tmp = tf.linalg.triangular_solve(tf.transpose(L), gamma, lower=False)
        mean = tf.linalg.matmul(w, tmp, transpose_a=True) + self.mean_function(Xnew)
        intermediateA = tf.linalg.triangular_solve(L, w, lower=True)

        if full_cov:
            var = (
                self.kernel(Xnew)
                - tf.linalg.matmul(w, w, transpose_a=True)
                + tf.linalg.matmul(intermediateA, intermediateA, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                - tf.reduce_sum(tf.square(w), 0)
                + tf.reduce_sum(tf.square(intermediateA), 0)
            )  # [N, P]
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean, var

class SGPR_deprecated(SGPRBase_deprecated):
    """
    Sparse GP regression.

    The key reference is :cite:t:`titsias2009variational`.

    For a use example see :doc:`../../../../notebooks/getting_started/large_data`.
    """

    class CommonTensors(NamedTuple):
        sigma_sq: tf.Tensor
        sigma: tf.Tensor
        A: tf.Tensor
        B: tf.Tensor
        LB: tf.Tensor
        AAT: tf.Tensor
        L: tf.Tensor

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    @check_shapes(
        "return.sigma_sq: [N]",
        "return.sigma: [N]",
        "return.A: [M, N]",
        "return.B: [M, M]",
        "return.LB: [M, M]",
        "return.AAT: [M, M]",
    )
    def _common_calculation(self) -> "SGPR.CommonTensors":
        """
        Matrices used in log-det calculation

        :return:
            * :math:`σ²`,
            * :math:`σ`,
            * :math:`A = L⁻¹K_{uf}/σ`, where :math:`LLᵀ = Kᵤᵤ`,
            * :math:`B = AAT+I`,
            * :math:`LB` where :math`LBLBᵀ = B`,
            * :math:`AAT = AAᵀ`,
        """
        x, _ = self.data  # [N]
        iv = self.inducing_variable  # [M]

        sigma_sq = tf.squeeze(self.likelihood.variance_at(x), axis=-1)  # [N]
        sigma = tf.sqrt(sigma_sq)  # [N]

        kuf = Kuf(iv, self.kernel, x)  # [M, N]
        kuu = Kuu(iv, self.kernel, jitter=default_jitter())  # [M, M]
        L = tf.linalg.cholesky(kuu)  # [M, M]

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = add_noise_cov(AAT, tf.cast(1.0, AAT.dtype))
        LB = tf.linalg.cholesky(B)

        return self.CommonTensors(sigma_sq, sigma, A, B, LB, AAT, L)

    @check_shapes(
        "return: []",
    )
    def logdet_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        r"""
        Bound from Jensen's Inequality:

        .. math::
            \log |K + σ²I| <= \log |Q + σ²I| + N * \log (1 + \textrm{tr}(K - Q)/(σ²N))

        :param common: A named tuple containing matrices that will be used
        :return: log_det, lower bound on :math:`-.5 * \textrm{output_dim} * \log |K + σ²I|`
        """
        sigma_sq = common.sigma_sq
        LB = common.LB
        AAT = common.AAT

        x, y = self.data
        outdim = to_default_float(tf.shape(y)[1])
        kdiag = self.kernel(x, full_cov=False)

        # tr(K) / σ²
        trace_k = tf.reduce_sum(kdiag / sigma_sq)
        # tr(Q) / σ²
        trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))
        # tr(K - Q) / σ²
        trace = trace_k - trace_q

        # 0.5 * log(det(B))
        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        # sum log(σ²)
        log_sigma_sq = tf.reduce_sum(tf.math.log(sigma_sq))

        logdet_k = -outdim * (half_logdet_b + 0.5 * log_sigma_sq + 0.5 * trace)
        return logdet_k

    @check_shapes(
        "return: []",
    )
    def quad_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        """
        :param common: A named tuple containing matrices that will be used
        :return: Lower bound on -.5 yᵀ(K + σ²I)⁻¹y
        """
        sigma = common.sigma
        A = common.A
        LB = common.LB

        x, y = self.data
        err = (y - self.mean_function(x)) / sigma[..., None]

        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)

        # σ⁻² yᵀy
        err_inner_prod = tf.reduce_sum(tf.square(err))
        c_inner_prod = tf.reduce_sum(tf.square(c))

        quad = -0.5 * (err_inner_prod - c_inner_prod)
        return quad

    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        common = self._common_calculation()
        output_shape = tf.shape(self.data[-1])
        num_data = to_default_float(output_shape[0])
        output_dim = to_default_float(output_shape[1])
        const = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        logdet = self.logdet_term(common)
        quad = self.quad_term(common)
        return const + logdet + quad

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        # could copy into posterior into a fused version

        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)

        sigma_sq = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)
        sigma = tf.sqrt(sigma_sq)

        L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
        A = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )  # cache qinv
        LB = tf.linalg.cholesky(B)  # cache alpha
        Aerr = tf.linalg.matmul(A, err / sigma[..., None])
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean + self.mean_function(Xnew), var

    @check_shapes(
        "return[0]: [M, P]",
        "return[1]: [M, M]",
    )
    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on
        inducing outputs.

        SVGP with this q(u) should predict identically to SGPR.

        :return: mu, cov
        """
        X_data, Y_data = self.data

        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        var = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)
        std = tf.sqrt(var)
        scaled_kuf = kuf / std
        sig = kuu + tf.matmul(scaled_kuf, scaled_kuf, transpose_b=True)
        sig_sqrt = tf.linalg.cholesky(sig)

        sig_sqrt_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)

        cov = tf.linalg.matmul(sig_sqrt_kuu, sig_sqrt_kuu, transpose_a=True)
        err = Y_data - self.mean_function(X_data)
        scaled_err = err / std[..., None]
        mu = tf.linalg.matmul(
            sig_sqrt_kuu,
            tf.linalg.triangular_solve(sig_sqrt, tf.linalg.matmul(scaled_kuf, scaled_err)),
            transpose_a=True,
        )

        return mu, cov

class SGPR(SGPR_with_posterior):
    # subclassed to ensure __class__ == "SGPR"

    __doc__ = SGPR_deprecated.__doc__  # Use documentation from SGPR_deprecated.

