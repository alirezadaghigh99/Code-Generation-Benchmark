class LinearCoregionalization(IndependentLatent, Combination):
    """
    Linear mixing of the latent GPs to form the output.
    """

    @check_shapes(
        "W: [P, L]",
    )
    def __init__(self, kernels: Sequence[Kernel], W: TensorType, name: Optional[str] = None):
        Combination.__init__(self, kernels=kernels, name=name)
        self.W = Parameter(W)

    @property
    def num_latent_gps(self) -> int:
        return self.W.shape[-1]  # type: ignore[no-any-return]  # L

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    @inherit_check_shapes
    def Kgg(self, X: TensorType, X2: TensorType) -> tf.Tensor:
        return cs(
            tf.stack([k.K(X, X2) for k in self.kernels], axis=0), "[L, batch..., N, batch2..., M]"
        )

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        Kxx = self.Kgg(X, X2)
        if X2 is None:
            cs(Kxx, "[L, batch..., N, N]")
            rank = tf.rank(X) - 1
            ones = tf.ones((rank + 1,), dtype=tf.int32)
            P = tf.shape(self.W)[0]
            L = tf.shape(self.W)[1]
            W_broadcast = cs(
                tf.reshape(self.W, tf.concat([[P, L], ones], 0)), "[P, L, broadcast batch..., 1, 1]"
            )
            KxxW = cs(Kxx[None, ...] * W_broadcast, "[P, L, batch..., N, N]")
            if full_output_cov:
                # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
                WKxxW = cs(tf.tensordot(self.W, KxxW, [[1], [1]]), "[P, P, batch..., N, N]")
                perm = tf.concat(
                    [
                        2 + tf.range(rank),
                        [0, 2 + rank, 1],
                    ],
                    0,
                )
                return cs(tf.transpose(WKxxW, perm), "[batch..., N, P, N, P]")
        else:
            cs(Kxx, "[L, batch..., N, batch2..., N2]")
            rank = tf.rank(X) - 1
            rank2 = tf.rank(X2) - 1
            ones12 = tf.ones((rank + rank2,), dtype=tf.int32)
            P = tf.shape(self.W)[0]
            L = tf.shape(self.W)[1]
            W_broadcast = cs(
                tf.reshape(self.W, tf.concat([[P, L], ones12], 0)),
                "[P, L, broadcast batch..., 1, broadcast batch2..., 1]",
            )
            KxxW = cs(Kxx[None, ...] * W_broadcast, "[P, L, batch..., N, batch2..., N2]")
            if full_output_cov:
                # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
                WKxxW = cs(
                    tf.tensordot(self.W, KxxW, [[1], [1]]), "[P, P, batch..., N, batch2..., N2]"
                )
                perm = tf.concat(
                    [
                        2 + tf.range(rank),
                        [0],
                        2 + rank + tf.range(rank2),
                        [1],
                    ],
                    0,
                )
                return cs(tf.transpose(WKxxW, perm), "[batch..., N, P, batch2..., N2, P]")
        # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
        return tf.reduce_sum(W_broadcast * KxxW, axis=1)

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        K = cs(tf.stack([k.K_diag(X) for k in self.kernels], axis=-1), "[batch..., N, L]")
        rank = tf.rank(X) - 1
        ones = tf.ones((rank,), dtype=tf.int32)

        if full_output_cov:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)
            Wt = cs(tf.transpose(self.W), "[L, P]")
            L = tf.shape(Wt)[0]
            P = tf.shape(Wt)[1]
            return cs(
                tf.reduce_sum(
                    cs(K[..., None, None], "[batch..., N, L, 1, 1]")
                    * cs(tf.reshape(Wt, tf.concat([ones, [L, P, 1]], 0)), "[..., L, P, 1]")
                    * cs(tf.reshape(Wt, tf.concat([ones, [L, 1, P]], 0)), "[..., L, 1, P]"),
                    axis=-3,
                ),
                "[batch..., N, P, P]",
            )
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)
            return cs(tf.linalg.matmul(K, self.W ** 2.0, transpose_b=True), "[batch..., N, P]")

class SharedIndependent(MultioutputKernel):
    """
    - Shared: we use the same kernel for each latent GP
    - Independent: Latents are uncorrelated a priori.

    .. warning::
       This class is created only for testing and comparison purposes.
       Use `gpflow.kernels` instead for more efficient code.
    """

    def __init__(self, kernel: Kernel, output_dim: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.output_dim = output_dim

    @property
    def num_latent_gps(self) -> int:
        # In this case number of latent GPs (L) == output_dim (P)
        return self.output_dim

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return (self.kernel,)

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        K = self.kernel.K(X, X2)
        rank = tf.rank(X) - 1
        if X2 is None:
            cs(K, "[batch..., N, N]")
            ones = tf.ones((rank,), dtype=tf.int32)
            if full_output_cov:
                multiples = tf.concat([ones, [1, self.output_dim]], 0)
                Ks = cs(tf.tile(K[..., None], multiples), "[batch..., N, N, P]")
                perm = tf.concat(
                    [
                        tf.range(rank),
                        [rank + 1, rank, rank + 2],
                    ],
                    0,
                )
                return cs(tf.transpose(tf.linalg.diag(Ks), perm), "[batch..., N, P, N, P]")
            else:
                multiples = tf.concat([[self.output_dim], ones, [1]], 0)
                return cs(tf.tile(K[None, ...], multiples), "[P, batch..., N, N]")

        else:
            cs(K, "[batch..., N, batch2..., N2]")
            rank2 = tf.rank(X2) - 1
            ones12 = tf.ones((rank + rank2,), dtype=tf.int32)
            if full_output_cov:
                multiples = tf.concat([ones12, [self.output_dim]], 0)
                Ks = cs(tf.tile(K[..., None], multiples), "[batch..., N, batch2..., N2, P]")
                perm = tf.concat(
                    [
                        tf.range(rank),
                        [rank + rank2],
                        rank + tf.range(rank2),
                        [rank + rank2 + 1],
                    ],
                    0,
                )
                return cs(
                    tf.transpose(tf.linalg.diag(Ks), perm), "[batch..., N, P, batch2..., N2, P]"
                )
            else:
                multiples = tf.concat([[self.output_dim], ones12], 0)
                return cs(tf.tile(K[None, ...], multiples), "[P, batch..., N, batch2..., N2]")

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        K = cs(self.kernel.K_diag(X), "[batch..., N]")
        rank = tf.rank(X) - 1
        ones = tf.ones((rank,), dtype=tf.int32)
        multiples = tf.concat([ones, [self.output_dim]], 0)
        Ks = cs(tf.tile(K[..., None], multiples), "[batch..., N, P]")
        return tf.linalg.diag(Ks) if full_output_cov else Ks

