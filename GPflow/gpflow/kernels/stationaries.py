class SquaredExponential(IsotropicStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(r) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    @inherit_check_shapes
    def K_r2(self, r2: TensorType) -> tf.Tensor:
        return self.variance * tf.exp(-0.5 * r2)

class Matern52(IsotropicStationary):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is

    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @check_shapes(
        "r: [batch..., N]",
        "return: [batch..., N]",
    )
    def K_r(self, r: TensorType) -> tf.Tensor:
        sqrt5 = np.sqrt(5.0)
        return self.variance * (1.0 + sqrt5 * r + 5.0 / 3.0 * tf.square(r)) * tf.exp(-sqrt5 * r)

class Exponential(IsotropicStationary):
    """
    The Exponential kernel. It is equivalent to a Matern12 kernel with doubled lengthscales.
    """

    @check_shapes(
        "r: [batch..., N]",
        "return: [batch..., N]",
    )
    def K_r(self, r: TensorType) -> tf.Tensor:
        return self.variance * tf.exp(-0.5 * r)

class Matern32(IsotropicStationary):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @check_shapes(
        "r: [batch..., N]",
        "return: [batch..., N]",
    )
    def K_r(self, r: TensorType) -> tf.Tensor:
        sqrt3 = np.sqrt(3.0)
        return self.variance * (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)

