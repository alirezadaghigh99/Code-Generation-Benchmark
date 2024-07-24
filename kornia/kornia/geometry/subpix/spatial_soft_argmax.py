class SpatialSoftArgmax2d(Module):
    r"""Compute the Spatial Soft-Argmax 2D of a given heatmap.

    See :func:`~kornia.geometry.subpix.spatial_soft_argmax2d` for details.
    """

    def __init__(self, temperature: Tensor = tensor(1.0), normalized_coordinates: bool = True) -> None:
        super().__init__()
        self.temperature: Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"temperature={self.temperature}, "
            f"normalized_coordinates={self.normalized_coordinates})"
        )

    def forward(self, input: Tensor) -> Tensor:
        return spatial_soft_argmax2d(input, self.temperature, self.normalized_coordinates)

