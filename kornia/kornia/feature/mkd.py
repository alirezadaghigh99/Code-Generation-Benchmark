class MKDDescriptor(nn.Module):
    r"""Module that computes Multiple Kernel local descriptors.

    This is based on the paper "Understanding and Improving Kernel Local Descriptors".
    See :cite:`mukundan2019understanding` for more details.

    Args:
        patch_size: Input patch size in pixels.
        kernel_type: Parametrization of kernel ``'concat'``, ``'cart'``, ``'polar'``.
        whitening: Whitening transform to apply ``None``, ``'lw'``, ``'pca'``, ``'pcawt'``, ``'pcaws'``.
        training_set: Set that model was trained on ``'liberty'``, ``'notredame'``, ``'yosemite'``.
        output_dims: Dimensionality reduction.

    Returns:
        Explicit cartesian or polar embedding.

    Shape:
        - Input: :math:`(B, in_{dims}, fmap_{size}, fmap_{size})`.
        - Output: :math:`(B, out_{dims}, fmap_{size}, fmap_{size})`,

    Examples:
        >>> patches = torch.rand(23, 1, 32, 32)
        >>> mkd = MKDDescriptor(patch_size=32,
        ...                     kernel_type='concat',
        ...                     whitening='pcawt',
        ...                     training_set='liberty',
        ...                     output_dims=128)
        >>> desc = mkd(patches) # 23x128
    """

    def __init__(
        self,
        patch_size: int = 32,
        kernel_type: str = "concat",
        whitening: str = "pcawt",
        training_set: str = "liberty",
        output_dims: int = 128,
    ) -> None:
        super().__init__()

        self.patch_size: int = patch_size
        self.kernel_type: str = kernel_type
        self.whitening: str = whitening
        self.training_set: str = training_set

        self.sigma = 1.4 * (patch_size / 64)
        self.smoothing = GaussianBlur2d((5, 5), (self.sigma, self.sigma), "replicate")
        self.gradients = MKDGradients()
        # This stupid thing needed for jitting...
        polar_s: str = "polar"
        cart_s: str = "cart"
        self.parametrizations = [polar_s, cart_s] if self.kernel_type == "concat" else [self.kernel_type]

        # Initialize cartesian/polar embedding with absolute/relative gradients.
        self.odims: int = 0
        relative_orientations = {polar_s: True, cart_s: False}
        self.feats = {}
        for parametrization in self.parametrizations:
            gradient_embedding = EmbedGradients(patch_size=patch_size, relative=relative_orientations[parametrization])
            spatial_encoding = ExplicitSpacialEncoding(
                kernel_type=parametrization, fmap_size=patch_size, in_dims=gradient_embedding.kernel.d
            )

            self.feats[parametrization] = nn.Sequential(gradient_embedding, spatial_encoding)
            self.odims += spatial_encoding.odims
        # Compute true output_dims.
        self.output_dims: int = min(output_dims, self.odims)

        # Load supervised(lw)/unsupervised(pca) model trained on training_set.
        if self.whitening is not None:
            whitening_models = torch.hub.load_state_dict_from_url(
                urls[self.kernel_type], map_location=map_location_to_cpu
            )
            whitening_model = whitening_models[training_set]
            self.whitening_layer = Whitening(
                whitening, whitening_model, in_dims=self.odims, output_dims=self.output_dims
            )
            self.odims = self.output_dims
        self.eval()

    def forward(self, patches: Tensor) -> Tensor:
        if not isinstance(patches, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(patches)}")
        if not len(patches.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect Bx1xHxW. Got: {patches.shape}")
        # Extract gradients.
        g = self.smoothing(patches)
        g = self.gradients(g)

        # Extract polar/cart features.
        features = []
        for parametrization in self.parametrizations:
            self.feats[parametrization].to(g.device)
            features.append(self.feats[parametrization](g))

        # Concatenate.
        y = torch.cat(features, dim=1)

        # l2-normalize.
        y = F.normalize(y, dim=1)

        # Whiten descriptors.
        if self.whitening is not None:
            y = self.whitening_layer(y)

        return y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"patch_size={self.patch_size}, "
            f"kernel_type={self.kernel_type}, "
            f"whitening={self.whitening}, "
            f"training_set={self.training_set}, "
            f"output_dims={self.output_dims})"
        )