class AbsorptionOnlyRaymarcher(torch.nn.Module):
    """
    Raymarch using the Absorption-Only (AO) algorithm.

    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., n_points_per_ray, 1)`, their values should range between [0, 1], and
    represent the opaqueness of each point (the higher the less transparent).
    The algorithm only measures the total amount of light absorbed along each ray
    and, besides outputting per-ray `opacity` values of shape `(...,)`,
    does not produce any feature renderings.

    The algorithm simply computes `total_transmission = prod(1 - rays_densities)`
    of shape `(..., 1)` which, for each ray, measures the total amount of light
    that passed through the volume.
    It then returns `opacities = 1 - total_transmission`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, rays_densities: torch.Tensor, **kwargs
    ) -> Union[None, torch.Tensor]:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray)` whose values range in [0, 1].

        Returns:
            opacities: A tensor of per-ray opacity values of shape `(..., 1)`.
                Its values range between [0, 1] and denote the total amount
                of light that has been absorbed for each ray. E.g. a value
                of 0 corresponds to the ray completely passing through a volume.
        """

        _check_raymarcher_inputs(
            rays_densities,
            None,
            None,
            features_can_be_none=True,
            z_can_be_none=True,
            density_1d=True,
        )
        rays_densities = rays_densities[..., 0]
        _check_density_bounds(rays_densities)
        total_transmission = torch.prod(1 - rays_densities, dim=-1, keepdim=True)
        opacities = 1.0 - total_transmission
        return opacities