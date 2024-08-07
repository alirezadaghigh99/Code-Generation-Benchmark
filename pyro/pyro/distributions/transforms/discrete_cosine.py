class DiscreteCosineTransform(Transform):
    """
    Discrete Cosine Transform of type-II.

    This uses :func:`~pyro.ops.tensor_utils.dct` and
    :func:`~pyro.ops.tensor_utils.idct` to compute
    orthonormal DCT and inverse DCT transforms. The jacobian is 1.

    :param int dim: Dimension along which to transform. Must be negative.
        This is an absolute dim counting from the right.
    :param float smooth: Smoothing parameter. When 0, this transforms white
        noise to white noise; when 1 this transforms Brownian noise to to white
        noise; when -1 this transforms violet noise to white noise; etc. Any
        real number is allowed. https://en.wikipedia.org/wiki/Colors_of_noise.
    """

    bijective = True

    def __init__(self, dim=-1, smooth=0.0, cache_size=0):
        assert isinstance(dim, int) and dim < 0
        self.dim = dim
        self.smooth = float(smooth)
        self._weight_cache = None
        super().__init__(cache_size=cache_size)

    def __hash__(self):
        return hash((type(self), self.dim, self.smooth))

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.dim == other.dim
            and self.smooth == other.smooth
        )

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(constraints.real, -self.dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(constraints.real, -self.dim)

    @torch.no_grad()
    def _weight(self, y):
        size = y.size(-1)
        if self._weight_cache is None or self._weight_cache.size(-1) != size:
            # Weight by frequency**smooth, where the DCT-II frequencies are:
            freq = torch.linspace(0.5, size - 0.5, size, dtype=y.dtype, device=y.device)
            w = freq.pow_(self.smooth)
            w /= w.log().mean().exp()  # Ensure |jacobian| = 1.
            self._weight_cache = w
        return self._weight_cache

    def _call(self, x):
        dim = self.dim
        if dim != -1:
            x = x.transpose(dim, -1)
        y = dct(x)
        if self.smooth:
            y = y * self._weight(y)
        if dim != -1:
            y = y.transpose(dim, -1)
        return y

    def _inverse(self, y):
        dim = self.dim
        if dim != -1:
            y = y.transpose(dim, -1)
        if self.smooth:
            y = y / self._weight(y)
        x = idct(y)
        if dim != -1:
            x = x.transpose(dim, -1)
        return x

    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros(x.shape[: self.dim])

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return DiscreteCosineTransform(self.dim, self.smooth, cache_size=cache_size)

    def forward_shape(self, shape):
        if len(shape) < self.event_dim:
            raise ValueError("Too few dimensions on input")
        return shape

    def inverse_shape(self, shape):
        if len(shape) < self.event_dim:
            raise ValueError("Too few dimensions on input")
        return shape

