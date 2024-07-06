def expectation(
    p: ProbabilityDistributionLike,
    obj1: PackedExpectationObject,
    obj2: PackedExpectationObject = None,
    nghp: Optional[int] = None,
) -> tf.Tensor:
    """
    Compute the expectation <obj1(x) obj2(x)>_p(x)
    Uses multiple-dispatch to select an analytical implementation,
    if one is available. If not, it falls back to quadrature.

    :type p: (mu, cov) tuple or a `ProbabilityDistribution` object
    :type obj1: kernel, mean function, (kernel, inducing_variable), or None
    :type obj2: kernel, mean function, (kernel, inducing_variable), or None
    :param int nghp: passed to `_quadrature_expectation` to set the number
                     of Gauss-Hermite points used: `num_gauss_hermite_points`
    :return: a 1-D, 2-D, or 3-D tensor containing the expectation

    Allowed combinations

    - Psi statistics:
        >>> eKdiag = expectation(p, kernel)  (N)  # Psi0
        >>> eKxz = expectation(p, (kernel, inducing_variable))  (NxM)  # Psi1
        >>> exKxz = expectation(p, identity_mean, (kernel, inducing_variable))  (NxDxM)
        >>> eKzxKxz = expectation(p, (kernel, inducing_variable), (kernel, inducing_variable))  (NxMxM)  # Psi2

    - kernels and mean functions:
        >>> eKzxMx = expectation(p, (kernel, inducing_variable), mean)  (NxMxQ)
        >>> eMxKxz = expectation(p, mean, (kernel, inducing_variable))  (NxQxM)

    - only mean functions:
        >>> eMx = expectation(p, mean)  (NxQ)
        >>> eM1x_M2x = expectation(p, mean1, mean2)  (NxQ1xQ2)
        .. note:: mean(x) is 1xQ (row vector)

    - different kernels. This occurs, for instance, when we are calculating Psi2 for Sum kernels:
        >>> eK1zxK2xz = expectation(p, (kern1, inducing_variable), (kern2, inducing_variable))  (NxMxM)
    """
    p, obj1, feat1, obj2, feat2 = _init_expectation(p, obj1, obj2)
    try:
        return dispatch.expectation(p, obj1, feat1, obj2, feat2, nghp=nghp)
    except NotImplementedError:
        return dispatch.quadrature_expectation(p, obj1, feat1, obj2, feat2, nghp=nghp)

