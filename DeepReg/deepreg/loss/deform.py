class BendingEnergy(tf.keras.layers.Layer):
    """
    Calculate the bending energy of ddf using central finite difference.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, name: str = "BendingEnergy", **kwargs):
        """
        Init.

        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = gradient_dxyz(ddf, gradient_dx)
        dfdy = gradient_dxyz(ddf, gradient_dy)
        dfdz = gradient_dxyz(ddf, gradient_dz)

        # second order gradient
        # (batch, m_dim1-4, m_dim2-4, m_dim3-4, 3)
        dfdxx = gradient_dxyz(dfdx, gradient_dx)
        dfdyy = gradient_dxyz(dfdy, gradient_dy)
        dfdzz = gradient_dxyz(dfdz, gradient_dz)
        dfdxy = gradient_dxyz(dfdx, gradient_dy)
        dfdyz = gradient_dxyz(dfdy, gradient_dz)
        dfdxz = gradient_dxyz(dfdx, gradient_dz)

        # (dx + dy + dz) ** 2 = dxx + dyy + dzz + 2*(dxy + dyz + dzx)
        energy = dfdxx ** 2 + dfdyy ** 2 + dfdzz ** 2
        energy += 2 * dfdxy ** 2 + 2 * dfdxz ** 2 + 2 * dfdyz ** 2
        return tf.reduce_mean(energy, axis=[1, 2, 3, 4])

