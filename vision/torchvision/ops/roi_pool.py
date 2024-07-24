class RoIPool(nn.Module):
    """
    See :func:`roi_pool`.
    """

    def __init__(self, output_size: BroadcastingList2[int], spatial_scale: float):
        super().__init__()
        _log_api_usage_once(self)
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input: Tensor, rois: Union[Tensor, List[Tensor]]) -> Tensor:
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(output_size={self.output_size}, spatial_scale={self.spatial_scale})"
        return s

