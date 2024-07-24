class CorrBlock1d(nn.Module):
    """The row-wise correlation block.

    Use indexes from correlation pyramid to create correlation features.
    The "indexing" of a given centroid pixel x' is done by concatenating its surrounding row neighbours
    within radius
    """

    def __init__(self, *, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.radius = radius
        self.out_channels = num_levels * (2 * radius + 1)

    def forward(self, centroids_coords: Tensor, corr_pyramid: List[Tensor]) -> Tensor:
        """Return correlation features by indexing from the pyramid."""
        neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        di = torch.linspace(-self.radius, self.radius, neighborhood_side_len, device=centroids_coords.device)
        di = di.view(1, 1, neighborhood_side_len, 1).to(centroids_coords.device)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2 but we only use the first one
        # We only consider 1d and take the first dim only
        centroids_coords = centroids_coords[:, :1].permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 1)

        indexed_pyramid = []
        for corr_volume in corr_pyramid:
            x0 = centroids_coords + di  # end shape is (batch_size * h * w, 1, side_len, 1)
            y0 = torch.zeros_like(x0)
            sampling_coords = torch.cat([x0, y0], dim=-1)
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode="bilinear").view(
                batch_size, h, w, -1
            )
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2

        corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

        expected_output_shape = (batch_size, self.out_channels, h, w)
        torch._assert(
            corr_features.shape == expected_output_shape,
            f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}",
        )
        return corr_features

class CorrPyramid1d(nn.Module):
    """Row-wise correlation pyramid.

    Create a row-wise correlation pyramid with ``num_levels`` level from the outputs of the feature encoder,
    this correlation pyramid will later be used as index to create correlation features using CorrBlock1d.
    """

    def __init__(self, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, fmap1: Tensor, fmap2: Tensor) -> List[Tensor]:
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2) on the same row.
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """

        torch._assert(
            fmap1.shape == fmap2.shape,
            f"Input feature maps should have the same shape, instead got {fmap1.shape} (fmap1.shape) != {fmap2.shape} (fmap2.shape)",
        )

        batch_size, num_channels, h, w = fmap1.shape
        fmap1 = fmap1.view(batch_size, num_channels, h, w)
        fmap2 = fmap2.view(batch_size, num_channels, h, w)

        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        corr = corr.view(batch_size, h, w, 1, w)
        corr_volume = corr / torch.sqrt(torch.tensor(num_channels, device=corr.device))

        corr_volume = corr_volume.reshape(batch_size * h * w, 1, 1, w)
        corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=(1, 2), stride=(1, 2))
            corr_pyramid.append(corr_volume)

        return corr_pyramid

