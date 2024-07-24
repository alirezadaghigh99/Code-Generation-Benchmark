class StudentT(ScalarLikelihood):
    def __init__(
        self,
        scale: ConstantOrFunction = 1.0,
        df: float = 3.0,
        scale_lower_bound: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param scale float: scale parameter
        :param df float: degrees of freedom
        """
        super().__init__(**kwargs)
        self.df = df
        self.scale_lower_bound = _lower_bound(scale_lower_bound)
        self.scale = prepare_parameter_or_function(scale, lower_bound=self.scale_lower_bound)

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [broadcast batch..., broadcast N, broadcast P]",
    )
    def _scale(self, X: TensorType) -> tf.Tensor:
        return evaluate_parameter_or_function(self.scale, X, lower_bound=self.scale_lower_bound)

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.student_t(Y, F, self._scale(X), self.df)

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return F

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        shape = tf.shape(F)
        var = (self._scale(X) ** 2) * (self.df / (self.df - 2.0))
        return tf.broadcast_to(var, shape)

