class IStandardKernel:
    """Interface for a standard kernel k(x, x') that in principle can be integrated.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IRBF`
       * :class:`emukit.quadrature.interfaces.IProductMatern52`
       * :class:`emukit.quadrature.interfaces.IProductMatern32`
       * :class:`emukit.quadrature.interfaces.IProductBrownian`

    """

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The kernel k(x1, x2) evaluated at x1 and x2.

        :param x1: First argument of the kernel, shape (n_points N, input_dim)
        :param x2: Second argument of the kernel, shape (n_points M, input_dim)
        :returns: Kernel evaluated at x1, x2, shape (N, M).
        """
        raise NotImplementedError

    # the following methods are gradients of the kernel
    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Gradient of the kernel wrt x1 evaluated at pair x1, x2.

        :param x1: First argument of the kernel, shape (n_points N, input_dim)
        :param x2: Second argument of the kernel, shape (n_points M, input_dim)
        :return: The gradient of the kernel wrt x1 evaluated at (x1, x2), shape (input_dim, N, M)
        """
        raise NotImplementedError

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """The gradient of the diagonal of the kernel (the variance) v(x):=k(x, x) evaluated at x.

        :param x: The locations where the gradient is evaluated, shape (n_points, input_dim).
        :return: The gradient of the diagonal of the kernel evaluated at x, shape (input_dim, n_points).
        """
        raise NotImplementedError

