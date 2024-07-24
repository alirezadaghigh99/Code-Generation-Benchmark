class Gaussian:
    """
    Non-normalized Gaussian distribution.

    This represents an arbitrary semidefinite quadratic function, which can be
    interpreted as a rank-deficient scaled Gaussian distribution. The precision
    matrix may have zero eigenvalues, thus it may be impossible to work
    directly with the covariance matrix.

    :param torch.Tensor log_normalizer: a normalization constant, which is mainly used to keep
        track of normalization terms during contractions.
    :param torch.Tensor info_vec: information vector, which is a scaled version of the mean
        ``info_vec = precision @ mean``. We use this represention to make gaussian contraction
        fast and stable.
    :param torch.Tensor precision: precision matrix of this gaussian.
    """

    def __init__(
        self,
        log_normalizer: torch.Tensor,
        info_vec: torch.Tensor,
        precision: torch.Tensor,
    ):
        # NB: using info_vec instead of mean to deal with rank-deficient problem
        assert info_vec.dim() >= 1
        assert precision.dim() >= 2
        assert precision.shape[-2:] == info_vec.shape[-1:] * 2
        self.log_normalizer = log_normalizer
        self.info_vec = info_vec
        self.precision = precision

    def dim(self):
        return self.info_vec.size(-1)

    @lazy_property
    def batch_shape(self):
        return broadcast_shape(
            self.log_normalizer.shape,
            self.info_vec.shape[:-1],
            self.precision.shape[:-2],
        )

    def expand(self, batch_shape) -> "Gaussian":
        n = self.dim()
        log_normalizer = self.log_normalizer.expand(batch_shape)
        info_vec = self.info_vec.expand(batch_shape + (n,))
        precision = self.precision.expand(batch_shape + (n, n))
        return Gaussian(log_normalizer, info_vec, precision)

    def reshape(self, batch_shape) -> "Gaussian":
        n = self.dim()
        log_normalizer = self.log_normalizer.reshape(batch_shape)
        info_vec = self.info_vec.reshape(batch_shape + (n,))
        precision = self.precision.reshape(batch_shape + (n, n))
        return Gaussian(log_normalizer, info_vec, precision)

    def __getitem__(self, index) -> "Gaussian":
        """
        Index into the batch_shape of a Gaussian.
        """
        assert isinstance(index, tuple)
        log_normalizer = self.log_normalizer[index]
        info_vec = self.info_vec[index + (slice(None),)]
        precision = self.precision[index + (slice(None), slice(None))]
        return Gaussian(log_normalizer, info_vec, precision)

    @staticmethod
    def cat(parts, dim=0) -> "Gaussian":
        """
        Concatenate a list of Gaussians along a given batch dimension.
        """
        if dim < 0:
            dim += len(parts[0].batch_shape)
        args = [
            torch.cat([getattr(g, attr) for g in parts], dim=dim)
            for attr in ["log_normalizer", "info_vec", "precision"]
        ]
        return Gaussian(*args)

    def event_pad(self, left=0, right=0) -> "Gaussian":
        """
        Pad along event dimension.
        """
        lr = (left, right)
        log_normalizer = self.log_normalizer
        info_vec = pad(self.info_vec, lr)
        precision = pad(self.precision, lr + lr)
        return Gaussian(log_normalizer, info_vec, precision)

    def event_permute(self, perm) -> "Gaussian":
        """
        Permute along event dimension.
        """
        assert isinstance(perm, torch.Tensor)
        assert perm.shape == (self.dim(),)
        info_vec = self.info_vec[..., perm]
        precision = self.precision[..., perm][..., perm, :]
        return Gaussian(self.log_normalizer, info_vec, precision)

    def __add__(self, other: Union["Gaussian", int, float, torch.Tensor]) -> "Gaussian":
        """
        Adds two Gaussians in log-density space.
        """
        if isinstance(other, Gaussian):
            assert self.dim() == other.dim()
            return Gaussian(
                self.log_normalizer + other.log_normalizer,
                self.info_vec + other.info_vec,
                self.precision + other.precision,
            )
        if isinstance(other, (int, float, torch.Tensor)):
            return Gaussian(self.log_normalizer + other, self.info_vec, self.precision)
        raise ValueError("Unsupported type: {}".format(type(other)))

    def __sub__(self, other: Union["Gaussian", int, float, torch.Tensor]) -> "Gaussian":
        if isinstance(other, (int, float, torch.Tensor)):
            return Gaussian(self.log_normalizer - other, self.info_vec, self.precision)
        raise ValueError("Unsupported type: {}".format(type(other)))

    def log_density(self, value: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log density of this Gaussian at a point value::

            -0.5 * value.T @ precision @ value + value.T @ info_vec + log_normalizer

        This is mainly used for testing.
        """
        if value.size(-1) == 0:
            batch_shape = broadcast_shape(value.shape[:-1], self.batch_shape)
            result: torch.Tensor = self.log_normalizer.expand(batch_shape)
            return result
        result = (-0.5) * matvecmul(self.precision, value)
        result = result + self.info_vec
        result = (value * result).sum(-1)
        return result + self.log_normalizer

    def rsample(
        self, sample_shape=torch.Size(), noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reparameterized sampler.
        """
        P_chol = safe_cholesky(self.precision)
        loc = self.info_vec.unsqueeze(-1).cholesky_solve(P_chol).squeeze(-1)
        shape = sample_shape + self.batch_shape + (self.dim(), 1)
        if noise is None:
            noise = torch.randn(shape, dtype=loc.dtype, device=loc.device)
        else:
            noise = noise.reshape(shape)
        noise = triangular_solve(noise, P_chol, upper=False, transpose=True).squeeze(-1)
        sample: torch.Tensor = loc + noise
        return sample

    def condition(self, value: torch.Tensor) -> "Gaussian":
        """
        Condition this Gaussian on a trailing subset of its state.
        This should satisfy::

            g.condition(y).dim() == g.dim() - y.size(-1)

        Note that since this is a non-normalized Gaussian, we include the
        density of ``y`` in the result. Thus :meth:`condition` is similar to a
        ``functools.partial`` binding of arguments::

            left = x[..., :n]
            right = x[..., n:]
            g.log_density(x) == g.condition(right).log_density(left)
        """
        assert isinstance(value, torch.Tensor)
        right = value.size(-1)
        dim = self.dim()
        assert right <= dim

        n = dim - right
        info_a = self.info_vec[..., :n]
        info_b = self.info_vec[..., n:]
        P_aa = self.precision[..., :n, :n]
        P_ab = self.precision[..., :n, n:]
        P_bb = self.precision[..., n:, n:]
        b = value

        info_vec = info_a - matvecmul(P_ab, b)
        precision = P_aa
        log_normalizer = (
            self.log_normalizer
            + -0.5 * matvecmul(P_bb, b).mul(b).sum(-1)
            + b.mul(info_b).sum(-1)
        )
        return Gaussian(log_normalizer, info_vec, precision)

    def left_condition(self, value: torch.Tensor) -> "Gaussian":
        """
        Condition this Gaussian on a leading subset of its state.
        This should satisfy::

            g.condition(y).dim() == g.dim() - y.size(-1)

        Note that since this is a non-normalized Gaussian, we include the
        density of ``y`` in the result. Thus :meth:`condition` is similar to a
        ``functools.partial`` binding of arguments::

            left = x[..., :n]
            right = x[..., n:]
            g.log_density(x) == g.left_condition(left).log_density(right)
        """
        assert isinstance(value, torch.Tensor)
        left = value.size(-1)
        dim = self.dim()
        assert left <= dim

        perm = torch.cat(
            [
                torch.arange(left, dim, device=value.device),
                torch.arange(left, device=value.device),
            ]
        )
        return self.event_permute(perm).condition(value)

    def marginalize(self, left=0, right=0) -> "Gaussian":
        """
        Marginalizing out variables on either side of the event dimension::

            g.marginalize(left=n).event_logsumexp() = g.logsumexp()
            g.marginalize(right=n).event_logsumexp() = g.logsumexp()

        and for data ``x``:

            g.condition(x).event_logsumexp()
              = g.marginalize(left=g.dim() - x.size(-1)).log_density(x)
        """
        if left == 0 and right == 0:
            return self
        if left > 0 and right > 0:
            raise NotImplementedError
        n = self.dim()
        n_b = left + right
        a = slice(left, n - right)  # preserved
        b = slice(None, left) if left else slice(n - right, None)

        P_aa = self.precision[..., a, a]
        P_ba = self.precision[..., b, a]
        P_bb = self.precision[..., b, b]
        P_b = safe_cholesky(P_bb)
        P_a = triangular_solve(P_ba, P_b, upper=False)
        P_at = P_a.transpose(-1, -2)
        precision = P_aa - matmul(P_at, P_a)

        info_a = self.info_vec[..., a]
        info_b = self.info_vec[..., b]
        b_tmp = triangular_solve(info_b.unsqueeze(-1), P_b, upper=False)
        info_vec = info_a - matmul(P_at, b_tmp).squeeze(-1)

        log_normalizer = (
            self.log_normalizer
            + 0.5 * n_b * math.log(2 * math.pi)
            - P_b.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            + 0.5 * b_tmp.squeeze(-1).pow(2).sum(-1)
        )
        return Gaussian(log_normalizer, info_vec, precision)

    def event_logsumexp(self) -> torch.Tensor:
        """
        Integrates out all latent state (i.e. operating on event dimensions).
        """
        n = self.dim()
        chol_P = safe_cholesky(self.precision)
        chol_P_u = triangular_solve(
            self.info_vec.unsqueeze(-1), chol_P, upper=False
        ).squeeze(-1)
        u_P_u = chol_P_u.pow(2).sum(-1)
        log_Z: torch.Tensor = (
            self.log_normalizer
            + 0.5 * n * math.log(2 * math.pi)
            + 0.5 * u_P_u
            - chol_P.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return log_Z

