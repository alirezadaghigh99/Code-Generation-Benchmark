class DensityFieldHead(FieldHead):
    """Density output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Softplus()) -> None:
        super().__init__(in_dim=in_dim, out_dim=1, field_head_name=FieldHeadNames.DENSITY, activation=activation)

class RGBFieldHead(FieldHead):
    """RGB output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Sigmoid()) -> None:
        super().__init__(in_dim=in_dim, out_dim=3, field_head_name=FieldHeadNames.RGB, activation=activation)

class SHFieldHead(FieldHead):
    """Spherical harmonics output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        levels: Number of spherical harmonics layers.
        channels: Number of channels. Defaults to 3 (ie RGB).
        activation: Output activation.
    """

    def __init__(
        self, in_dim: Optional[int] = None, levels: int = 3, channels: int = 3, activation: Optional[nn.Module] = None
    ) -> None:
        out_dim = channels * levels**2
        super().__init__(in_dim=in_dim, out_dim=out_dim, field_head_name=FieldHeadNames.SH, activation=activation)

