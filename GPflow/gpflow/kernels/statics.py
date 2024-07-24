class White(Static):
    """
    The White kernel: this kernel produces 'white noise'. The kernel equation is

        k(x_n, x_m) = δ(n, m) σ²

    where:
    δ(.,.) is the Kronecker delta,
    σ²  is the variance parameter.
    """

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        if X2 is None:
            d = tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
            return tf.linalg.diag(d)
        else:
            shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], axis=0)
            return tf.zeros(shape, dtype=X.dtype)

