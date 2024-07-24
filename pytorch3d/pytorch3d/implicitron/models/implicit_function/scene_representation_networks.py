class SRNPixelGenerator(Configurable, torch.nn.Module):
    n_harmonic_functions: int = 4
    n_hidden_units: int = 256
    n_hidden_units_color: int = 128
    n_layers: int = 2
    in_features: int = 256
    out_features: int = 3
    ray_dir_in_camera_coords: bool = False

    def __post_init__(self):
        self._harmonic_embedding = HarmonicEmbedding(
            self.n_harmonic_functions, append_input=True
        )
        self._net = pytorch_prototyping.FCBlock(
            hidden_ch=self.n_hidden_units,
            num_hidden_layers=self.n_layers,
            in_features=self.in_features,
            out_features=self.n_hidden_units,
        )
        self._density_layer = torch.nn.Linear(self.n_hidden_units, 1)
        self._density_layer.apply(_kaiming_normal_init)
        embedding_dim_dir = self._harmonic_embedding.get_output_dim(input_dims=3)
        self._color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                self.n_hidden_units + embedding_dim_dir,
                self.n_hidden_units_color,
            ),
            torch.nn.LayerNorm([self.n_hidden_units_color]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.n_hidden_units_color, self.out_features),
        )
        self._color_layer.apply(_kaiming_normal_init)

    # TODO: merge with NeuralRadianceFieldBase's _get_colors
    def _get_colors(self, features: torch.Tensor, rays_directions: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.net`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self._harmonic_embedding(rays_directions_normed)
        return self._color_layer((features, rays_embedding))

    def forward(
        self,
        raymarch_features: torch.Tensor,
        ray_bundle: ImplicitronRayBundle,
        camera: Optional[CamerasBase] = None,
        **kwargs,
    ):
        """
        Args:
            raymarch_features: Features from the raymarching network of shape
                `(minibatch, ..., self.in_features)`
            ray_bundle: An ImplicitronRayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacitiy of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # raymarch_features.shape = [minibatch x ... x pts_per_ray x 3]
        features = self._net(raymarch_features)
        # features.shape = [minibatch x ... x self.n_hidden_units]

        if self.ray_dir_in_camera_coords:
            if camera is None:
                raise ValueError("Camera must be given if xyz_ray_dir_in_camera_coords")

            directions = ray_bundle.directions @ camera.R
        else:
            directions = ray_bundle.directions

        # NNs operate on the flattenned rays; reshaping to the correct spatial size
        features = features.reshape(*raymarch_features.shape[:-1], -1)

        raw_densities = self._density_layer(features)

        rays_colors = self._get_colors(features, directions)

        return raw_densities, rays_colors

class SRNImplicitFunction(ImplicitFunctionBase, torch.nn.Module):
    latent_dim: int = 0
    raymarch_function: SRNRaymarchFunction
    pixel_generator: SRNPixelGenerator

    def __post_init__(self):
        run_auto_creation(self)

    def create_raymarch_function(self) -> None:
        self.raymarch_function = SRNRaymarchFunction(
            latent_dim=self.latent_dim,
            **self.raymarch_function_args,
        )

    @classmethod
    def raymarch_function_tweak_args(cls, type, args: DictConfig) -> None:
        args.pop("latent_dim", None)

    def forward(
        self,
        *,
        ray_bundle: ImplicitronRayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        raymarch_features: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        predict_colors = raymarch_features is not None
        if predict_colors:
            return self.pixel_generator(
                raymarch_features=raymarch_features,
                ray_bundle=ray_bundle,
                camera=camera,
                **kwargs,
            )
        else:
            return self.raymarch_function(
                ray_bundle=ray_bundle,
                fun_viewpool=fun_viewpool,
                camera=camera,
                global_code=global_code,
                **kwargs,
            )

class SRNHyperNetImplicitFunction(ImplicitFunctionBase, torch.nn.Module):
    """
    This implicit function uses a hypernetwork to generate the
    SRNRaymarchingFunction, and this is cached. Whenever the
    global_code changes, `on_bind_args` must be called to clear
    the cache.
    """

    latent_dim_hypernet: int = 0
    latent_dim: int = 0
    hypernet: SRNRaymarchHyperNet
    pixel_generator: SRNPixelGenerator

    def __post_init__(self):
        run_auto_creation(self)

    def create_hypernet(self) -> None:
        self.hypernet = SRNRaymarchHyperNet(
            latent_dim=self.latent_dim,
            latent_dim_hypernet=self.latent_dim_hypernet,
            **self.hypernet_args,
        )

    @classmethod
    def hypernet_tweak_args(cls, type, args: DictConfig) -> None:
        args.pop("latent_dim", None)
        args.pop("latent_dim_hypernet", None)

    def forward(
        self,
        *,
        ray_bundle: ImplicitronRayBundle,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        raymarch_features: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        predict_colors = raymarch_features is not None
        if predict_colors:
            return self.pixel_generator(
                raymarch_features=raymarch_features,
                ray_bundle=ray_bundle,
                camera=camera,
                **kwargs,
            )
        else:
            return self.hypernet(
                ray_bundle=ray_bundle,
                fun_viewpool=fun_viewpool,
                camera=camera,
                global_code=global_code,
                **kwargs,
            )

    def on_bind_args(self):
        """
        The global_code may have changed, so we reset the hypernet.
        """
        self.hypernet.cached_srn_raymarch_function = None

