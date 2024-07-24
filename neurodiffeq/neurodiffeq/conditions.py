class DirichletBVP(BaseCondition):
    r"""A double-ended Dirichlet boundary condition:
    :math:`u(t_0)=u_0` and :math:`u(t_1)=u_1`.

    :param t_0: The initial time.
    :type t_0: float
    :param u_0: The initial value of :math:`u`. :math:`u(t_0)=u_0`.
    :type u_0: float
    :param t_1: The final time.
    :type t_1: float
    :param u_1: The initial value of :math:`u`. :math:`u(t_1)=u_1`.
    :type u_1: float
    """

    @deprecated_alias(x_0='u_0', x_1='u_1')
    def __init__(self, t_0, u_0, t_1, u_1):
        super().__init__()
        self.t_0, self.u_0, self.t_1, self.u_1 = t_0, u_0, t_1, u_1

    def parameterize(self, output_tensor, t):
        r"""Re-parameterizes outputs such that the Dirichlet condition is satisfied on both ends of the domain.

        The re-parameterization is
        :math:`\displaystyle u(t)=(1-\tilde{t})u_0+\tilde{t}u_1+\left(1-e^{(1-\tilde{t})\tilde{t}}\right)\mathrm{ANN}(t)`,
        where :math:`\displaystyle \tilde{t} = \frac{t-t_0}{t_1-t_0}` and :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param t: Input to the neural network; i.e., sampled time-points or another independent variable.
        :type t: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """

        t_tilde = (t - self.t_0) / (self.t_1 - self.t_0)
        return self.u_0 * (1 - t_tilde) \
               + self.u_1 * t_tilde \
               + (1 - torch.exp((1 - t_tilde) * t_tilde)) * output_tensor

class BundleIVP(BaseCondition, _BundleConditionMixin):
    r"""An initial value problem of one of the following forms:

    - Dirichlet condition: :math:`u(t_0,\boldsymbol{\theta})=u_0`.
    - Neumann condition: :math:`\displaystyle\frac{\partial u}{\partial t}\bigg|_{t = t_0}(\boldsymbol{\theta}) = u_0'`.

    Here :math:`\boldsymbol{\theta}=(\theta_{1},\theta_{2},...,\theta_{n})\in\mathbb{R}^n`,
    where each :math:`\theta_i` represents a parameter, or a condition, of the ODE system that we want to solve.

    :param t_0: The initial time. Ignored if present in `bundle_param_lookup`.
    :type t_0: float
    :param u_0:
        The initial value of :math:`u`. :math:`u(t_0,\boldsymbol{\theta})=u_0`.
        Ignored if present in `bundle_param_lookup`.
    :type u_0: float
    :param u_0_prime:
        The initial derivative of :math:`u` w.r.t. :math:`t`.
        :math:`\displaystyle\frac{\partial u}{\partial t}\bigg|_{t = t_0}(\boldsymbol{\theta}) = u_0'`.
        Defaults to None.
        Ignored if present in `bundle_param_lookup`.
    :type u_0_prime: float, optional
    :param bundle_param_lookup: See _BundleConditionMixin for details. Allowed keys are 't_0', 'u_0', and 'u_0_prime'.
    :type bundle_param_lookup: Dict[str, int]
    """

    @deprecated_alias(x_0='u_0', x_0_prime='u_0_prime', bundle_conditions='bundle_param_lookup')
    def __init__(self, t_0=None, u_0=None, u_0_prime=None, bundle_param_lookup=None):
        BaseCondition.__init__(self)
        _BundleConditionMixin.__init__(
            self,
            bundle_param_lookup=bundle_param_lookup,
            allowed_params=['t_0', 'u_0', 'u_0_prime'],
        )
        self.t_0, self.u_0, self.u_0_prime = t_0, u_0, u_0_prime

    def parameterize(self, output_tensor, t, *theta):
        r"""Re-parameterizes outputs such that the Dirichlet/Neumann condition is satisfied.

        if t_0 is not included in the bundle:

        - For Dirichlet condition, the re-parameterization is
          :math:`\displaystyle u(t,\boldsymbol{\theta}) = u_0 + \left(1 - e^{-(t-t_0)}\right)`
          :math:`\mathrm{ANN}(t,\boldsymbol{\theta})`
        - For Neumann condition, the re-parameterization is
          :math:`\displaystyle u(t,\boldsymbol{\theta}) = u_0 + (t-t_0) u'_0 + \left(1 - e^{-(t-t_0)}\right)^2`
          :math:`\mathrm{ANN}(t,\boldsymbol{\theta})`

        if t_0 is included in the bundle:

        - For Dirichlet condition, the re-parameterization is
          :math:`\displaystyle u(t,\boldsymbol{\theta}) = u_0 + \left(t - t_0\right)`
          :math:`\mathrm{ANN}(t,\boldsymbol{\theta})`
        - For Neumann condition, the re-parameterization is
          :math:`\displaystyle u(t,\boldsymbol{\theta}) = u_0 + (t-t_0) u'_0 + \left(t - t_0\right)^2`
          :math:`\mathrm{ANN}(t,\boldsymbol{\theta})`

        Where :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param t: First input to the neural network; i.e., sampled time-points; i.e., independent variables.
        :type t: `torch.Tensor`
        :param theta: Rest of the inputs to the neural network; i.e., sampled bundle-points
        :type theta: tuple[torch.Tensor, ..., torch.Tensor]
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """

        t_0 = self._get_parameter('t_0', theta)
        u_0 = self._get_parameter('u_0', theta)
        u_0_prime = self._get_parameter('u_0_prime', theta)

        if u_0_prime is None:
            return u_0 + (1 - torch.exp(-t + t_0)) * output_tensor
        else:
            return u_0 + (t - t_0) * u_0_prime + ((1 - torch.exp(-t + t_0)) ** 2) * output_tensor

class BundleDirichletBVP(BaseCondition, _BundleConditionMixin):
    r"""A double-ended Dirichlet boundary condition: :math:`u(t_0)=u_0` and :math:`u(t_1)=u_1`.

    :param t_0: The initial time. Ignored if 't_0' is present in bundle_param_lookup.
    :type t_0: float
    :param u_0: The initial value of :math:`u`. :math:`u(t_0)=u_0`. Ignored if 'u_0' is present in bundle_param_lookup.
    :type u_0: float
    :param t_1: The final time. Ignored if 't_1' is present in bundle_param_lookup.
    :type t_1: float
    :param u_1: The initial value of :math:`u`. :math:`u(t_1)=u_1`. Ignored if 'u_1' is present in bundle_param_lookup.
    :type u_1: float
    :param bundle_param_lookup: See _BundleConditionMixin for details. Allowed keys are 't_0', 'u_0', 't_1', and 'u_1'.
    :type bundle_param_lookup: Dict[str, int]
    """

    @deprecated_alias(bundle_conditions='bundle_param_lookup')
    def __init__(self, t_0, u_0, t_1, u_1, bundle_param_lookup=None):
        BaseCondition.__init__(self)
        _BundleConditionMixin.__init__(
            self,
            bundle_param_lookup=bundle_param_lookup,
            allowed_params=['t_0', 'u_0', 't_1', 'u_1'],
        )
        self.t_0, self.u_0, self.t_1, self.u_1 = t_0, u_0, t_1, u_1

    def parameterize(self, output_tensor, t, *theta):
        r"""Re-parameterizes outputs such that the Dirichlet condition is satisfied on both ends of the domain.

        The re-parameterization is
        :math:`\displaystyle u(t)=(1-\tilde{t})u_0+\tilde{t}u_1+\left(1-e^{(1-\tilde{t})\tilde{t}}\right)\mathrm{ANN}(t)`,
        where :math:`\displaystyle \tilde{t} = \frac{t-t_0}{t_1-t_0}` and :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param t: Input to the neural network; i.e., sampled time-points or another independent variable.
        :type t: `torch.Tensor`
        :param theta: Bundle parameters that potentially override default parameters.
        :type theta: Tuple[`torch.Tensor`]
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """
        u_0 = self._get_parameter('u_0', theta)
        u_1 = self._get_parameter('u_1', theta)
        t_0 = self._get_parameter('t_0', theta)
        t_1 = self._get_parameter('t_1', theta)

        t_tilde = (t - t_0) / (t_1 - t_0)
        return u_0 * (1 - t_tilde) + u_1 * t_tilde + (1 - torch.exp((1 - t_tilde) * t_tilde)) * output_tensor

class IVP(BaseCondition):
    r"""An initial value problem of one of the following forms:

    - Dirichlet condition: :math:`u(t_0)=u_0`.
    - Neumann condition: :math:`\displaystyle\frac{\partial u}{\partial t}\bigg|_{t = t_0} = u_0'`.

    :param t_0: The initial time.
    :type t_0: float
    :param u_0: The initial value of :math:`u`. :math:`u(t_0)=u_0`.
    :type u_0: float
    :param u_0_prime:
        The initial derivative of :math:`u` w.r.t. :math:`t`.
        :math:`\displaystyle\frac{\partial u}{\partial t}\bigg|_{t = t_0} = u_0'`.
        Defaults to None.
    :type u_0_prime: float, optional
    """

    @deprecated_alias(x_0='u_0', x_0_prime='u_0_prime')
    def __init__(self, t_0, u_0=None, u_0_prime=None):
        super().__init__()
        self.t_0, self.u_0, self.u_0_prime = t_0, u_0, u_0_prime

    def parameterize(self, output_tensor, t):
        r"""Re-parameterizes outputs such that the Dirichlet/Neumann condition is satisfied.

        - For Dirichlet condition, the re-parameterization is
          :math:`\displaystyle u(t) = u_0 + \left(1 - e^{-(t-t_0)}\right) \mathrm{ANN}(t)`
          where :math:`\mathrm{ANN}` is the neural network.
        - For Neumann condition, the re-parameterization is
          :math:`\displaystyle u(t) = u_0 + (t-t_0) u'_0 + \left(1 - e^{-(t-t_0)}\right)^2 \mathrm{ANN}(t)`
          where :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param t: Input to the neural network; i.e., sampled time-points; i.e., independent variables.
        :type t: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """
        if self.u_0_prime is None:
            return self.u_0 + (1 - torch.exp(-t + self.t_0)) * output_tensor
        else:
            return self.u_0 + (t - self.t_0) * self.u_0_prime + ((1 - torch.exp(-t + self.t_0)) ** 2) * output_tensor

class DirichletBVPSpherical(BaseCondition):
    r"""The Dirichlet boundary condition for the interior and exterior boundary of the sphere,
    where the interior boundary is not necessarily a point. The conditions are:

    - :math:`u(r_0,\theta,\phi)=f(\theta,\phi)`
    - :math:`u(r_1,\theta,\phi)=g(\theta,\phi)`

    :param r_0:
        The radius of the interior boundary.
        When :math:`r_0 = 0`, the interior boundary collapses to a single point (center of the ball).
    :type r_0: float
    :param f:
        The value of :math:`u` on the interior boundary.
        :math:`u(r_0, \theta, \phi)=f(\theta, \phi)`.
    :type f: callable
    :param r_1:
        The radius of the exterior boundary.
        If set to None, `g` must also be None.
    :type r_1: float or None
    :param g:
        The value of :math:`u` on the exterior boundary.
        :math:`u(r_1, \theta, \phi)=g(\theta, \phi)`.
        If set to None, `r_1` must also be set to None.
    :type g: callable or None
    """

    def __init__(self, r_0, f, r_1=None, g=None):
        super(DirichletBVPSpherical, self).__init__()
        if (r_1 is None) ^ (g is None):
            raise ValueError(f'r_1 and g must be both/neither set to None; got r_1={r_1}, g={g}')
        self.r_0, self.r_1 = r_0, r_1
        self.f, self.g = f, g

    def parameterize(self, output_tensor, r, theta, phi):
        r"""Re-parameterizes outputs such that the Dirichlet condition is satisfied on both spherical boundaries.

        - If both inner and outer boundaries are specified
          :math:`u(r_0,\theta,\phi)=f(\theta,\phi)` and
          :math:`u(r_1,\theta,\phi)=g(\theta,\phi)`:

          The re-parameterization is
          :math:`\big(1-\tilde{r}\big)f(\theta,\phi)+\tilde{r}g(\theta,\phi)
          +\Big(1-e^{\tilde{r}(1-{\tilde{r}})}\Big)\mathrm{ANN}(r, \theta, \phi)`
          where :math:`\displaystyle\tilde{r}=\frac{r-r_0}{r_1-r_0}`;

        - If only one boundary is specified (inner or outer) :math:`u(r_0,\theta,\phi)=f(\theta,\phi)`

          The re-parameterization is
          :math:`f(\theta,\phi)+\Big(1-e^{-|r-r_0|}\Big)\mathrm{ANN}(r, \theta, \phi)`;

        where :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param r: The radii (or :math:`r`-component) of the inputs to the network.
        :type r: `torch.Tensor`
        :param theta: The co-latitudes (or :math:`\theta`-component) of the inputs to the network.
        :type theta: `torch.Tensor`
        :param phi: The longitudes (or :math:`\phi`-component) of the inputs to the network.
        :type phi: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """
        if self.r_1 is None:
            return (1 - torch.exp(-torch.abs(r - self.r_0))) * output_tensor + self.f(theta, phi)
        else:
            r_tilde = (r - self.r_0) / (self.r_1 - self.r_0)
            return self.f(theta, phi) * (1 - r_tilde) + \
                   self.g(theta, phi) * r_tilde + \
                   (1. - torch.exp((1 - r_tilde) * r_tilde)) * output_tensor

class DirichletBVP2D(BaseCondition):
    r"""An Dirichlet boundary condition on the boundary of :math:`[x_0, x_1] \times [y_0, y_1]`, where

    - :math:`u(x_0, y) = f_0(y)`;
    - :math:`u(x_1, y) = f_1(y)`;
    - :math:`u(x, y_0) = g_0(x)`;
    - :math:`u(x, y_1) = g_1(x)`.

    :param x_min: The lower bound of x, the :math:`x_0`.
    :type x_min: float
    :param x_min_val: The boundary value on :math:`x = x_0`, i.e. :math:`f_0(y)`.
    :type x_min_val: callable
    :param x_max: The upper bound of x, the :math:`x_1`.
    :type x_max: float
    :param x_max_val: The boundary value on :math:`x = x_1`, i.e. :math:`f_1(y)`.
    :type x_max_val: callable
    :param y_min: The lower bound of y, the :math:`y_0`.
    :type y_min: float
    :param y_min_val: The boundary value on :math:`y = y_0`, i.e. :math:`g_0(x)`.
    :type y_min_val: callable
    :param y_max: The upper bound of y, the :math:`y_1`.
    :type y_max: float
    :param y_max_val: The boundary value on :math:`y = y_1`, i.e. :math:`g_1(x)`.
    :type y_max_val: callable
    """

    def __init__(self, x_min, x_min_val, x_max, x_max_val, y_min, y_min_val, y_max, y_max_val):
        r"""Initializer method
        """
        super().__init__()
        self.x0, self.f0 = x_min, x_min_val
        self.x1, self.f1 = x_max, x_max_val
        self.y0, self.g0 = y_min, y_min_val
        self.y1, self.g1 = y_max, y_max_val

    def parameterize(self, output_tensor, x, y):
        r"""Re-parameterizes outputs such that the Dirichlet condition is satisfied on all four sides of the domain.

        The re-parameterization is
        :math:`\displaystyle u(x,y)=A(x,y)
        +\tilde{x}\big(1-\tilde{x}\big)\tilde{y}\big(1-\tilde{y}\big)\mathrm{ANN}(x,y)`, where

        :math:`\displaystyle \begin{align*}
        A(x,y)=&\big(1-\tilde{x}\big)f_0(y)+\tilde{x}f_1(y) \\
        &+\big(1-\tilde{y}\big)\Big(g_0(x)-\big(1-\tilde{x}\big)g_0(x_0)+\tilde{x}g_0(x_1)\Big) \\
        &+\tilde{y}\Big(g_1(x)-\big(1-\tilde{x}\big)g_1(x_0)+\tilde{x}g_1(x_1)\Big)
        \end{align*}`

        :math:`\displaystyle\tilde{x}=\frac{x-x_0}{x_1-x_0}`,

        :math:`\displaystyle\tilde{y}=\frac{y-y_0}{y_1-y_0}`,

        and :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param x: :math:`x`-coordinates of inputs to the neural network; i.e., the sampled :math:`x`-coordinates.
        :type x: `torch.Tensor`
        :param y: :math:`y`-coordinates of inputs to the neural network; i.e., the sampled :math:`y`-coordinates.
        :type y: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """
        x_tilde = (x - self.x0) / (self.x1 - self.x0)
        y_tilde = (y - self.y0) / (self.y1 - self.y0)
        x0 = torch.ones_like(x_tilde[0, 0]).expand(*x_tilde.shape) * self.x0
        x1 = torch.ones_like(x_tilde[0, 0]).expand(*x_tilde.shape) * self.x1
        Axy = (1 - x_tilde) * self.f0(y) + x_tilde * self.f1(y) \
              + (1 - y_tilde) * (self.g0(x) - ((1 - x_tilde) * self.g0(x0) + x_tilde * self.g0(x1))) \
              + y_tilde * (self.g1(x) - ((1 - x_tilde) * self.g1(x0) + x_tilde * self.g1(x1)))

        return Axy + x_tilde * (1 - x_tilde) * y_tilde * (1 - y_tilde) * output_tensor

class NoCondition(BaseCondition):
    r"""A polymorphic condition where no re-parameterization will be performed.

    .. note::
        This condition is called *polymorphic* because it can be enforced on networks of arbitrary input/output sizes.
    """

    def parameterize(self, output_tensor, *input_tensors):
        r"""Performs no re-parameterization, or identity parameterization, in this case.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param input_tensors: Inputs to the neural network; i.e., sampled coordinates; i.e., independent variables.
        :type input_tensors: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """
        return output_tensor

