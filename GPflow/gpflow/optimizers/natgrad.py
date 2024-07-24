class NaturalGradient(tf_keras.optimizers.Optimizer):
    """
    Implements a natural gradient descent optimizer for variational models
    that are based on a distribution q(u) = N(q_mu, q_sqrt q_sqrtᵀ) that is
    parameterized by mean q_mu and lower-triangular Cholesky factor q_sqrt
    of the covariance.

    Note that this optimizer does not implement the standard API of
    tf_keras.optimizers.Optimizer. Its only public method is minimize(), which has
    a custom signature (var_list needs to be a list of (q_mu, q_sqrt) tuples,
    where q_mu and q_sqrt are gpflow.Parameter instances, not tf.Variable).

    Note furthermore that the natural gradients are implemented only for the
    full covariance case (i.e., q_diag=True is NOT supported).

    When using in your work, please cite :cite:t:`salimbeni18`.
    """

    def __init__(
        self, gamma: Scalar, xi_transform: XiTransform = XiNat(), name: Optional[str] = None
    ) -> None:
        """
        :param gamma: natgrad step length
        :param xi_transform: default ξ transform (can be overridden in the call to minimize())
            The XiNat default choice works well in general.
        """
        name = self.__class__.__name__ if name is None else name
        super().__init__(name)
        # explicitly store name (as TF <2.12 stores it differently from TF 2.12+)
        self.natgrad_name = name
        self.gamma = gamma
        self.xi_transform = xi_transform

    @check_shapes(
        "var_list[all][0]: [N, D]",
        "var_list[all][1]: [D, N, N]",
    )
    def minimize(
        self,
        loss_fn: LossClosure,
        var_list: Sequence[NatGradParameters],
    ) -> None:
        """
        Minimizes objective function of the model.
        Natural Gradient optimizer works with variational parameters only.

        GPflow implements the `XiNat` (default) and `XiSqrtMeanVar` transformations
        for parameters. Custom transformations that implement the `XiTransform`
        interface are also possible.

        :param loss_fn: Loss function.
        :param var_list: List of pair tuples of variational parameters or
            triplet tuple with variational parameters and ξ transformation.
            If ξ is not specified, will use self.xi_transform.
            For example, `var_list` could be::

                var_list = [
                    (q_mu1, q_sqrt1),
                    (q_mu2, q_sqrt2, XiSqrtMeanVar())
                ]
        """
        parameters = [(v[0], v[1], (v[2] if len(v) > 2 else None)) for v in var_list]  # type: ignore[misc]
        self._natgrad_steps(loss_fn, parameters)

    @check_shapes(
        "parameters[all][0]: [N, D]",
        "parameters[all][1]: [D, N, N]",
    )
    def _natgrad_steps(
        self,
        loss_fn: LossClosure,
        parameters: Sequence[Tuple[Parameter, Parameter, Optional[XiTransform]]],
    ) -> None:
        """
        Computes gradients of loss_fn() w.r.t. q_mu and q_sqrt, and updates
        these parameters using the natgrad backwards step, for all sets of
        variational parameters passed in.

        :param loss_fn: Loss function.
        :param parameters: List of tuples (q_mu, q_sqrt, xi_transform)
        """
        q_mus, q_sqrts, xis = zip(*parameters)
        q_mu_vars = [p.unconstrained_variable for p in q_mus]
        q_sqrt_vars = [p.unconstrained_variable for p in q_sqrts]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(q_mu_vars + q_sqrt_vars)
            loss = loss_fn()

        q_mu_grads, q_sqrt_grads = tape.gradient(loss, [q_mu_vars, q_sqrt_vars])
        # NOTE that these are the gradients in *unconstrained* space

        with tf.name_scope(f"{self.natgrad_name}/natural_gradient_steps"):
            for q_mu_grad, q_sqrt_grad, q_mu, q_sqrt, xi_transform in zip(
                q_mu_grads, q_sqrt_grads, q_mus, q_sqrts, xis
            ):
                self._natgrad_apply_gradients(q_mu_grad, q_sqrt_grad, q_mu, q_sqrt, xi_transform)

    @check_shapes(
        "q_mu_grad: [N, D]",
        "q_sqrt_grad: [D, N_N_transformed...]",
        "q_mu: [N, D]",
        "q_sqrt: [D, N, N]",
    )
    def _natgrad_apply_gradients(
        self,
        q_mu_grad: tf.Tensor,
        q_sqrt_grad: tf.Tensor,
        q_mu: Parameter,
        q_sqrt: Parameter,
        xi_transform: Optional[XiTransform] = None,
    ) -> None:
        """
        This function does the backward step on the q_mu and q_sqrt parameters,
        given the gradients of the loss function with respect to their unconstrained
        variables. I.e., it expects the arguments to come from

            with tf.GradientTape() as tape:
                loss = loss_function()
            q_mu_grad, q_mu_sqrt = tape.gradient(loss, [q_mu, q_sqrt])

        (Note that tape.gradient() returns the gradients in *unconstrained* space!)

        Implements equation [10] from :cite:t:`salimbeni18`.

        In addition, for convenience with the rest of GPflow, this code computes ∂L/∂η using
        the chain rule (the following assumes a numerator layout where the gradient is a row
        vector; note that TensorFlow actually returns a column vector), where L is the loss:

        ∂L/∂η = (∂L / ∂[q_mu, q_sqrt])(∂[q_mu, q_sqrt] / ∂η)

        In total there are three derivative calculations:
        natgrad of L w.r.t ξ  = (∂ξ / ∂θ) [(∂L / ∂[q_mu, q_sqrt]) (∂[q_mu, q_sqrt] / ∂η)]ᵀ

        Note that if ξ = θ (i.e. [q_mu, q_sqrt]) some of these calculations are the identity.
        In the code η = eta, ξ = xi, θ = nat.

        :param q_mu_grad: gradient of loss w.r.t. q_mu (in unconstrained space)
        :param q_sqrt_grad: gradient of loss w.r.t. q_sqrt (in unconstrained space)
        :param q_mu: parameter for the mean of q(u) with shape [M, L]
        :param q_sqrt: parameter for the square root of the covariance of q(u)
            with shape [L, M, M] (the diagonal parametrization, q_diag=True, is NOT supported)
        :param xi_transform: the ξ transform to use (self.xi_transform if not specified)
        """
        if xi_transform is None:
            xi_transform = self.xi_transform

        # 1) the ordinary gpflow gradient
        dL_dmean = _to_constrained(q_mu_grad, q_mu.transform)
        dL_dvarsqrt = _to_constrained(q_sqrt_grad, q_sqrt.transform)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch([q_mu.unconstrained_variable, q_sqrt.unconstrained_variable])

            # the three parameterizations as functions of [q_mu, q_sqrt]
            eta1, eta2 = meanvarsqrt_to_expectation(q_mu, q_sqrt)
            # we need these to calculate the relevant gradients
            meanvarsqrt = expectation_to_meanvarsqrt(eta1, eta2)

            if not isinstance(xi_transform, XiNat):
                nat1, nat2 = meanvarsqrt_to_natural(q_mu, q_sqrt)
                xi1_nat, xi2_nat = xi_transform.naturals_to_xi(nat1, nat2)
                dummy_tensors = tf.ones_like(xi1_nat), tf.ones_like(xi2_nat)
                with tf.GradientTape(watch_accessed_variables=False) as forward_tape:
                    forward_tape.watch(dummy_tensors)
                    dummy_gradients = tape.gradient(
                        [xi1_nat, xi2_nat], [nat1, nat2], output_gradients=dummy_tensors
                    )

        # 2) the chain rule to get ∂L/∂η, where η (eta) are the expectation parameters
        dL_deta1, dL_deta2 = tape.gradient(
            meanvarsqrt, [eta1, eta2], output_gradients=[dL_dmean, dL_dvarsqrt]
        )

        if not isinstance(xi_transform, XiNat):
            nat_dL_xi1, nat_dL_xi2 = forward_tape.gradient(
                dummy_gradients, dummy_tensors, output_gradients=[dL_deta1, dL_deta2]
            )
        else:
            nat_dL_xi1, nat_dL_xi2 = dL_deta1, dL_deta2

        del tape  # Remove "persistent" tape

        xi1, xi2 = xi_transform.meanvarsqrt_to_xi(q_mu, q_sqrt)
        xi1_new = xi1 - self.gamma * nat_dL_xi1
        xi2_new = xi2 - self.gamma * nat_dL_xi2

        # Transform back to the model parameters [q_mu, q_sqrt]
        mean_new, varsqrt_new = xi_transform.xi_to_meanvarsqrt(xi1_new, xi2_new)

        q_mu.assign(mean_new)
        q_sqrt.assign(varsqrt_new)

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config()
        config.update({"gamma": self._serialize_hyperparameter("gamma")})
        return config

class XiSqrtMeanVar(XiTransform):
    """
    This transformation will perform natural gradient descent on the model parameters,
    so saves the conversion to and from Xi.
    """

    @staticmethod
    @check_shapes(
        "mean: [N, D]",
        "varsqrt: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def meanvarsqrt_to_xi(mean: tf.Tensor, varsqrt: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return mean, varsqrt

    @staticmethod
    @check_shapes(
        "xi1: [N, D]",
        "xi2: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def xi_to_meanvarsqrt(xi1: tf.Tensor, xi2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return xi1, xi2

    @staticmethod
    @check_shapes(
        "nat1: [N, D]",
        "nat2: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def naturals_to_xi(nat1: tf.Tensor, nat2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return natural_to_meanvarsqrt(nat1, nat2)

