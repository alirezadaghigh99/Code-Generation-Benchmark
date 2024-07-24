class Polynomial(MeanFunction, Function):
    """
    A generic polynomial mean function.
    """

    @check_shapes("w: [broadcast output_dim, broadcast n_terms]")
    def __init__(
        self, degree: int, input_dim: int = 1, output_dim: int = 1, w: Optional[TensorType] = None
    ) -> None:
        """
        :param degree: The degree of the polynomial.
        :param input_dim: Number of inputs / variables this polynomial is defined over.
        :param output_dim: Number of outputs / polynomials.
        :param w: Initial weights of the terms of the polynomial. The inner dimension (``n_terms``)
            should correspond to the powers returned by ``compute_powers``.
        """
        powers = cs(tuple(self.compute_powers(degree, input_dim)), "[n_terms, input_dim]")
        if w is None:
            w = cs([1.0] + (len(powers) - 1) * [0.0], "[n_terms]")
        w_shape = (output_dim, len(powers))
        self.powers = tf.constant(powers, dtype=default_float())
        self.w = Parameter(tf.broadcast_to(w, w_shape))

    @staticmethod
    def compute_powers(degree: int, input_dim: int) -> Sequence[Tuple[int, ...]]:
        """
        Computes integer tuples corresponding to the powers to raise inputs to.

        Specifically this returns, in lexicographical order, all tuples where:

        * The tuple has length `input_dim`.
        * The values are non-negative integers.
        * The sum of the tuple is no greater than `degree`.

        For example::

            compute_powers(degree=2, input_dim=3)

        returns::

            (0, 0, 0)
            (0, 0, 1)
            (0, 0, 2)
            (0, 1, 0)
            (0, 1, 1)
            (0, 2, 0)
            (1, 0, 0)
            (1, 0, 1)
            (1, 1, 0)
            (2, 0, 0)

        where a tuple::

            (1, 0, 2)

        will translate to a the term::

            w[i] * (x[0]**1) * (x[1]**0) * (x[2]**2)
        """
        if not input_dim:
            return [()]
        result = []
        for i in range(degree + 1):
            for inner in Polynomial.compute_powers(degree - i, input_dim - 1):
                result.append((i,) + inner)
        return result

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        raised = cs(tf.pow(X[..., None, :], self.powers), "[batch..., n_terms, input_dim]")
        prod = cs(tf.math.reduce_prod(raised, axis=-1), "[batch..., n_terms]")
        return tf.einsum("...i,ji->...j", prod, self.w)

class Linear(MeanFunction, Function):
    """
    y_i = A x_i + b
    """

    @check_shapes(
        "A: [broadcast D, broadcast Q]",
        "b: [broadcast Q]",
    )
    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        if isinstance(A, Parameter):
            if len(A._shape) >= 2:
                self.A = A
            else:
                raise ValueError(
                    "Error 'gpflow.funcitons.Linear()' mean function. A has not the correct shape (at least 2d)."
                )
        else:
            self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.tensordot(X, self.A, [[-1], [0]]) + self.b

class Additive(MeanFunction, Function):
    def __init__(self, first_part: Function, second_part: Function) -> None:
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.add(self.add_1(X), self.add_2(X))

class Linear(MeanFunction, Function):
    """
    y_i = A x_i + b
    """

    @check_shapes(
        "A: [broadcast D, broadcast Q]",
        "b: [broadcast Q]",
    )
    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        if isinstance(A, Parameter):
            if len(A._shape) >= 2:
                self.A = A
            else:
                raise ValueError(
                    "Error 'gpflow.funcitons.Linear()' mean function. A has not the correct shape (at least 2d)."
                )
        else:
            self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.tensordot(X, self.A, [[-1], [0]]) + self.b

class Zero(Constant, Function):
    def __init__(self, output_dim: int = 1) -> None:
        Constant.__init__(self)
        self.output_dim = output_dim
        del self.c

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        output_shape = tf.concat([tf.shape(X)[:-1], [self.output_dim]], axis=0)
        return tf.zeros(output_shape, dtype=X.dtype)

class Product(MeanFunction, Function):
    def __init__(self, first_part: Function, second_part: Function):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.multiply(self.prod_1(X), self.prod_2(X))

class Product(MeanFunction, Function):
    def __init__(self, first_part: Function, second_part: Function):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.multiply(self.prod_1(X), self.prod_2(X))

