class PSRoIAlign(nn.Module):
    """
    See :func:`ps_roi_align`.
    """

    def __init__(
        self,
        output_size: int,
        spatial_scale: float,
        sampling_ratio: int,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return ps_roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"output_size={self.output_size}"
            f", spatial_scale={self.spatial_scale}"
            f", sampling_ratio={self.sampling_ratio}"
            f")"
        )
        return s

