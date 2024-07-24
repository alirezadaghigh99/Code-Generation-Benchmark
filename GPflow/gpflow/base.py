class Parameter(tfp.util.TransformedVariable):
    """A parameter retains both constrained and unconstrained representations. If no transform
    is provided, these two values will be the same.  It is often challenging for humans to operate with
    unconstrained parameters, although this is typically easier for the optimiser. For example, a variance cannot be negative, therefore we need a
    positive constraint and it is natural to use constrained values.  A prior can be imposed
    either on the constrained version (default) or on the unconstrained version of the
    parameter.

    See `this guide <https://gpflow.github.io/GPflow/develop/notebooks/getting_started/parameters_and_their_optimisation.html#The-Module-and-Parameter-classes>`_
    for an introduction to this class.

    :param unconstrained_shape: Declare the shape of the unconstrained / pre-transformed values.
        Useful for setting dynamic shapes.
    :param constrained_shape: Declare the shape of the constrained / transformed values. Useful
        for setting dynamic shapes.
    :param shape: Convenience shortcut for setting both `unconstrained_shape` and
        `constrained_shape` to the same value.
    """

    def __init__(
        self,
        value: "TensorData",
        *,
        transform: Optional[Transform] = None,
        prior: Optional[Prior] = None,
        prior_on: Optional[Union[str, PriorOn]] = None,
        trainable: Optional[bool] = None,
        dtype: Optional[DType] = None,
        name: Optional[str] = None,
        unconstrained_shape: Optional[Sequence[Optional[int]]] = None,
        constrained_shape: Optional[Sequence[Optional[int]]] = None,
        shape: Optional[Sequence[Optional[int]]] = None,
    ):

        if transform:
            name = name or transform.name

        if isinstance(value, Parameter):
            transform = transform or value.transform
            prior = prior or value.prior
            prior_on = prior_on or value.prior_on
            name = name or value.bijector.name
            trainable = value.trainable if trainable is None else trainable

            if dtype:
                tensor_value: TensorType = _cast_to_dtype(value, dtype)
            else:
                tensor_value = value
        else:
            if transform is None:
                transform = tfp.bijectors.Identity()

            prior_on = prior_on if prior_on else PriorOn.CONSTRAINED
            trainable = trainable if trainable is not None else True

            tensor_value = _cast_to_dtype(value, dtype)

        _validate_unconstrained_value(tensor_value, transform, dtype)

        if shape is not None:
            assert unconstrained_shape is None, "Cannot set both `shape` and `unconstrained_shape`."
            assert constrained_shape is None, "Cannot set both `shape` and `constrained_shape`."
            unconstrained_shape = shape
            constrained_shape = shape

        super().__init__(
            tensor_value,
            transform,
            dtype=tensor_value.dtype,
            trainable=trainable,
            name=name,
            shape=unconstrained_shape,
        )

        # TransformedVariable.__init__ doesn't allow us to pass an unconstrained / pre-transformed
        # shape, so we manually override it.
        if constrained_shape is not None:
            self._shape = tf.TensorShape(constrained_shape)

        self.prior: Optional[Prior] = prior
        self.prior_on = prior_on  # type: ignore[assignment]  # see https://github.com/python/mypy/issues/3004

    @check_shapes("return: []")
    def log_prior_density(self) -> tf.Tensor:
        """ Log of the prior probability density of the constrained variable. """

        if self.prior is None:
            return tf.convert_to_tensor(0.0, dtype=self.dtype)

        y = self

        if self.prior_on == PriorOn.CONSTRAINED:
            # evaluation is in same space as prior
            return tf.reduce_sum(self.prior.log_prob(y))

        else:
            # prior on unconstrained, but evaluating log-prior in constrained space
            x = self.unconstrained_variable
            log_p = tf.reduce_sum(self.prior.log_prob(x))

            if self.transform is not None:
                # need to include log|Jacobian| to account for coordinate transform
                log_det_jacobian = self.transform.inverse_log_det_jacobian(y, y.shape.ndims)
                log_p += tf.reduce_sum(log_det_jacobian)

            return log_p

    @property
    def prior_on(self) -> PriorOn:
        return self._prior_on

    @prior_on.setter
    def prior_on(self, value: Union[str, PriorOn]) -> None:
        self._prior_on = PriorOn(value)

    @property
    def unconstrained_variable(self) -> tf.Variable:
        return self._pretransformed_input

    @property
    def transform(self) -> Optional[Transform]:
        return self.bijector

    @property
    def trainable(self) -> bool:
        """
        `True` if this instance is trainable, else `False`.

        This attribute cannot be set directly. Use :func:`gpflow.set_trainable`.
        """
        return self.unconstrained_variable.trainable  # type: ignore[no-any-return]

    def assign(
        self,
        value: "TensorData",
        use_locking: bool = False,
        name: Optional[str] = None,
        read_value: bool = True,
    ) -> tf.Tensor:
        """
        Assigns constrained `value` to the unconstrained parameter's variable.
        It passes constrained value through parameter's transform first.

        Example::

            a = Parameter(2.0, transform=tfp.bijectors.Softplus())
            b = Parameter(3.0)

            a.assign(4.0)               # `a` parameter to `2.0` value.
            a.assign(tf.constant(5.0))  # `a` parameter to `5.0` value.
            a.assign(b)                 # `a` parameter to constrained value of `b`.


        :param value: Constrained tensor-like value.
        :param use_locking: If `True`, use locking during the assignment.
        :param name: The name of the operation to be created.
        :param read_value: if True, will return something which evaluates to the new
            value of the variable; if False will return the assign op.
        """
        unconstrained_value = _validate_unconstrained_value(value, self.transform, self.dtype)
        return self.unconstrained_variable.assign(
            unconstrained_value, use_locking=use_locking, name=name, read_value=read_value
        )

class Parameter(tfp.util.TransformedVariable):
    """A parameter retains both constrained and unconstrained representations. If no transform
    is provided, these two values will be the same.  It is often challenging for humans to operate with
    unconstrained parameters, although this is typically easier for the optimiser. For example, a variance cannot be negative, therefore we need a
    positive constraint and it is natural to use constrained values.  A prior can be imposed
    either on the constrained version (default) or on the unconstrained version of the
    parameter.

    See `this guide <https://gpflow.github.io/GPflow/develop/notebooks/getting_started/parameters_and_their_optimisation.html#The-Module-and-Parameter-classes>`_
    for an introduction to this class.

    :param unconstrained_shape: Declare the shape of the unconstrained / pre-transformed values.
        Useful for setting dynamic shapes.
    :param constrained_shape: Declare the shape of the constrained / transformed values. Useful
        for setting dynamic shapes.
    :param shape: Convenience shortcut for setting both `unconstrained_shape` and
        `constrained_shape` to the same value.
    """

    def __init__(
        self,
        value: "TensorData",
        *,
        transform: Optional[Transform] = None,
        prior: Optional[Prior] = None,
        prior_on: Optional[Union[str, PriorOn]] = None,
        trainable: Optional[bool] = None,
        dtype: Optional[DType] = None,
        name: Optional[str] = None,
        unconstrained_shape: Optional[Sequence[Optional[int]]] = None,
        constrained_shape: Optional[Sequence[Optional[int]]] = None,
        shape: Optional[Sequence[Optional[int]]] = None,
    ):

        if transform:
            name = name or transform.name

        if isinstance(value, Parameter):
            transform = transform or value.transform
            prior = prior or value.prior
            prior_on = prior_on or value.prior_on
            name = name or value.bijector.name
            trainable = value.trainable if trainable is None else trainable

            if dtype:
                tensor_value: TensorType = _cast_to_dtype(value, dtype)
            else:
                tensor_value = value
        else:
            if transform is None:
                transform = tfp.bijectors.Identity()

            prior_on = prior_on if prior_on else PriorOn.CONSTRAINED
            trainable = trainable if trainable is not None else True

            tensor_value = _cast_to_dtype(value, dtype)

        _validate_unconstrained_value(tensor_value, transform, dtype)

        if shape is not None:
            assert unconstrained_shape is None, "Cannot set both `shape` and `unconstrained_shape`."
            assert constrained_shape is None, "Cannot set both `shape` and `constrained_shape`."
            unconstrained_shape = shape
            constrained_shape = shape

        super().__init__(
            tensor_value,
            transform,
            dtype=tensor_value.dtype,
            trainable=trainable,
            name=name,
            shape=unconstrained_shape,
        )

        # TransformedVariable.__init__ doesn't allow us to pass an unconstrained / pre-transformed
        # shape, so we manually override it.
        if constrained_shape is not None:
            self._shape = tf.TensorShape(constrained_shape)

        self.prior: Optional[Prior] = prior
        self.prior_on = prior_on  # type: ignore[assignment]  # see https://github.com/python/mypy/issues/3004

    @check_shapes("return: []")
    def log_prior_density(self) -> tf.Tensor:
        """ Log of the prior probability density of the constrained variable. """

        if self.prior is None:
            return tf.convert_to_tensor(0.0, dtype=self.dtype)

        y = self

        if self.prior_on == PriorOn.CONSTRAINED:
            # evaluation is in same space as prior
            return tf.reduce_sum(self.prior.log_prob(y))

        else:
            # prior on unconstrained, but evaluating log-prior in constrained space
            x = self.unconstrained_variable
            log_p = tf.reduce_sum(self.prior.log_prob(x))

            if self.transform is not None:
                # need to include log|Jacobian| to account for coordinate transform
                log_det_jacobian = self.transform.inverse_log_det_jacobian(y, y.shape.ndims)
                log_p += tf.reduce_sum(log_det_jacobian)

            return log_p

    @property
    def prior_on(self) -> PriorOn:
        return self._prior_on

    @prior_on.setter
    def prior_on(self, value: Union[str, PriorOn]) -> None:
        self._prior_on = PriorOn(value)

    @property
    def unconstrained_variable(self) -> tf.Variable:
        return self._pretransformed_input

    @property
    def transform(self) -> Optional[Transform]:
        return self.bijector

    @property
    def trainable(self) -> bool:
        """
        `True` if this instance is trainable, else `False`.

        This attribute cannot be set directly. Use :func:`gpflow.set_trainable`.
        """
        return self.unconstrained_variable.trainable  # type: ignore[no-any-return]

    def assign(
        self,
        value: "TensorData",
        use_locking: bool = False,
        name: Optional[str] = None,
        read_value: bool = True,
    ) -> tf.Tensor:
        """
        Assigns constrained `value` to the unconstrained parameter's variable.
        It passes constrained value through parameter's transform first.

        Example::

            a = Parameter(2.0, transform=tfp.bijectors.Softplus())
            b = Parameter(3.0)

            a.assign(4.0)               # `a` parameter to `2.0` value.
            a.assign(tf.constant(5.0))  # `a` parameter to `5.0` value.
            a.assign(b)                 # `a` parameter to constrained value of `b`.


        :param value: Constrained tensor-like value.
        :param use_locking: If `True`, use locking during the assignment.
        :param name: The name of the operation to be created.
        :param read_value: if True, will return something which evaluates to the new
            value of the variable; if False will return the assign op.
        """
        unconstrained_value = _validate_unconstrained_value(value, self.transform, self.dtype)
        return self.unconstrained_variable.assign(
            unconstrained_value, use_locking=use_locking, name=name, read_value=read_value
        )

