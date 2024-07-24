class RobustMax(Module):
    r"""
    This class represent a multi-class inverse-link function. Given a vector
    :math:`f=[f_1, f_2, ... f_k]`, the result of the mapping is

    .. math::

       y = [y_1 ... y_k]

    with

    .. math::

       y_i = \left\{
       \begin{array}{ll}
           (1-\varepsilon)   & \textrm{if} \ i = \textrm{argmax}(f) \\
           \varepsilon/(k-1) & \textrm{otherwise}
       \end{array}
       \right.

    where :math:`k` is the number of classes.
    """

    @check_shapes(
        "epsilon: []",
    )
    def __init__(self, num_classes: int, epsilon: float = 1e-3, **kwargs: Any) -> None:
        """
        `epsilon` represents the fraction of 'errors' in the labels of the
        dataset. This may be a hard parameter to optimize, so by default
        it is set un-trainable, at a small value.
        """
        super().__init__(**kwargs)
        transform = tfp.bijectors.Sigmoid()
        prior = tfp.distributions.Beta(to_default_float(0.2), to_default_float(5.0))
        self.epsilon = Parameter(epsilon, transform=transform, prior=prior, trainable=False)
        self.num_classes = num_classes
        self._squash = 1e-6

    @check_shapes(
        "F: [broadcast batch..., latent_dim]",
        "return: [batch..., latent_dim]",
    )
    def __call__(self, F: TensorType) -> tf.Tensor:
        i = tf.argmax(F, 1)
        return tf.one_hot(i, self.num_classes, 1.0 - self.epsilon, self.eps_k1)

    @property  # type: ignore[misc]  # Mypy doesn't like decorated properties.
    @check_shapes(
        "return: []",
    )
    def eps_k1(self) -> tf.Tensor:
        return self.epsilon / (self.num_classes - 1.0)

    @check_shapes(
        "val: [batch...]",
        "return: [batch...]",
    )
    def safe_sqrt(self, val: TensorType) -> tf.Tensor:
        return tf.sqrt(tf.maximum(val, 1e-10))

    @check_shapes(
        "Y: [broadcast batch..., observation_dim]",
        "mu: [broadcast batch..., latent_dim]",
        "var: [broadcast batch..., latent_dim]",
        "gh_x: [n_quad_points]",
        "gh_w: [n_quad_points]",
        "return: [batch..., observation_dim]",
    )
    def prob_is_largest(
        self, Y: TensorType, mu: TensorType, var: TensorType, gh_x: TensorType, gh_w: TensorType
    ) -> tf.Tensor:
        Y = to_default_int(Y)
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(
            tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1.0, 0.0), dtype=mu.dtype
        )
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            self.safe_sqrt(2.0 * var_selected), (-1, 1)
        )

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            self.safe_sqrt(var), 2
        )
        cdfs = 0.5 * (1.0 + tf.math.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2 * self._squash) + self._squash

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(
            tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0.0, 1.0), dtype=mu.dtype
        )
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.reduce_prod(cdfs, axis=[1]) @ tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1))

