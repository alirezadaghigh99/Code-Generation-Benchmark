class ZonalSphericalHarmonics(FunctionBasis):
    """Zonal harmonics (spherical harmonics with order=0)

    :param max_degree: highest degrees to be included; degrees will contain {0, 1, ..., max_degree}; ignored if `degrees` is passed
    :type max_degree: int
    :param degrees: a list of degrees to be used, must be nonnegative and unique; if passed, `max_degrees` will be ignored
    :type degrees: list[int]
    """

    def __init__(self, max_degree=None, degrees=None):
        if max_degree is None and degrees is None:
            raise ValueError("Either `max_degree` or `degrees` must be specified")
        if max_degree is not None and degrees is not None:
            warnings.warn(f"degrees={degrees} specified, ignoring max_degree={max_degree}")

        self.max_degree = max_degree
        if degrees is None:
            degrees = list(range(max_degree + 1))
        self.degrees = degrees

        coefficients = [np.sqrt((2 * l + 1) / (4 * np.pi)) for l in self.degrees]
        polynomials = [LegendrePolynomial(d) for d in self.degrees]

        # The `c=c` and `fn=fn` in the lambda is needed due this issue:
        # https://stackoverflow.com/questions/28268439/python-list-comprehension-with-lambdas
        fns = [
            lambda theta, c=c, fn=fn: fn(torch.cos(theta)) * c
            for c, fn in zip(coefficients, polynomials)
        ]
        self.basis_module = CustomBasis(fns)

    def __call__(self, theta, phi):
        return self.basis_module(theta)

class RealSphericalHarmonics(FunctionBasis):
    """Spherical harmonics as a function basis

    :param max_degree: highest degree (currently only supports l<=4) for the spherical harmonics_fn
    :type max_degree: int
    """

    def __init__(self, max_degree=4):
        super(RealSphericalHarmonics, self).__init__()
        self.harmonics = []
        self.max_degree = max_degree
        if max_degree >= 0:
            self.harmonics += [Y0_0]
        if max_degree >= 1:
            self.harmonics += [Y1n1, Y1_0, Y1p1]
        if max_degree >= 2:
            self.harmonics += [Y2n2, Y2n1, Y2_0, Y2p1, Y2p2]
        if max_degree >= 3:
            self.harmonics += [Y3n3, Y3n2, Y3n1, Y3_0, Y3p1, Y3p2, Y3p3]
        if max_degree >= 4:
            self.harmonics += [Y4n4, Y4n3, Y4n2, Y4n1, Y4_0, Y4p1, Y4p2, Y4p3, Y4p4]
        if max_degree >= 5:
            raise NotImplementedError(f'max_degree = {max_degree} not implemented for {self.__class__.__name__} yet')

    def __call__(self, theta, phi):
        """Compute the value of each spherical harmonic component evaluated at each point.

        :param theta: theta in spherical coordinates, must have shape (-1, 1)
        :type theta: `torch.Tensor`
        :param phi: phis in spherical coordinates, must have the same shape as theta
        :type phi: `torch.Tensor`
        :return: spherical harmonics evaluated at each point, will be of shape (-1, n_components)
        :rtype: `torch.Tensor`
        """
        if len(theta.shape) != 2 or theta.shape[1] != 1:
            raise ValueError(f'theta must be of shape (-1, 1); got {theta.shape}')
        if theta.shape != phi.shape:
            raise ValueError(f'theta/phi must be of the same shape; got f{theta.shape} and f{phi.shape}')
        components = [Y(theta, phi) for Y in self.harmonics]
        return torch.cat(components, dim=1)

class LegendrePolynomial:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = legendre(degree)

    def __call__(self, x):
        if self.degree == 0:
            return torch.ones_like(x, requires_grad=x.requires_grad)
        elif self.degree == 1:
            return x * 1
        else:
            return sum(coeff * x ** (self.degree - i) for i, coeff in enumerate(self.coefficients))

class LegendreBasis(FunctionBasis):
    def __init__(self, max_degree):
        polynomials = [LegendrePolynomial(d) for d in range(max_degree + 1)]
        self.basis_module = CustomBasis(polynomials)

    def __call__(self, x):
        return self.basis_module(x)

