def _compute_splatting_colors_and_weights(
    pixel_coords_screen: torch.Tensor,
    colors: torch.Tensor,
    sigma: float,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """
    For each center pixel q, compute the splatting weights of its surrounding nine spla-
    tting pixels p, as well as their splatting colors (which are just their colors re-
    weighted by the splatting weights).

    Args:
        pixel_coords_screen: (N, H, W, K, 2) tensor of pixel screen coords.
        colors: (N, H, W, K, 4) RGBA tensor of pixel colors.
        sigma: splatting kernel variance.
        offsets: (9, 2) tensor computed by _precompute, indicating the nine
            splatting directions ([-1, -1], ..., [1, 1]).

    Returns:
        splat_colors_and_weights: (N, H, W, K, 9, 5) tensor.
            splat_colors_and_weights[..., :4] corresponds to the splatting colors, and
            splat_colors_and_weights[..., 4:5] to the splatting weights. The "9" di-
            mension corresponds to the nine splatting directions.
    """
    N, H, W, K, C = colors.shape
    splat_kernel_normalization = _get_splat_kernel_normalization(offsets, sigma)

    # Distance from each barycentric-interpolated triangle vertices' triplet from its
    # "ideal" pixel-center location. pixel_coords_screen are in screen coordinates, and
    # should be at the "ideal" locations on the forward pass -- e.g.
    # pixel_coords_screen[n, 24, 31, k] = [24.5, 31.5]. For this reason, q_to_px_center
    # should equal torch.zeros during the forward pass. On the backwards pass, these
    # coordinates will be adjusted and non-zero, allowing the gradients to flow back
    # to the mesh vertex coordinates.
    q_to_px_center = (
        torch.floor(pixel_coords_screen[..., :2]) - pixel_coords_screen[..., :2] + 0.5
    ).view((N, H, W, K, 1, 2))

    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    dist2_p_q = torch.sum((q_to_px_center + offsets) ** 2, dim=5)  # (N, H, W, K, 9)
    splat_weights = torch.exp(-dist2_p_q / (2 * sigma**2))
    alpha = colors[..., 3:4]
    splat_weights = (alpha * splat_kernel_normalization * splat_weights).unsqueeze(
        5
    )  # (N, H, W, K, 9, 1)

    # splat_colors[n, h, w, direction, :] contains the splatting color (weighted by the
    # splatting weight) that pixel h, w will splat in one  of the nine possible
    # directions (e.g. nhw0 corresponds to splatting in [-1, 1] direciton, nhw4 is
    # self-splatting).
    splat_colors = splat_weights * colors.unsqueeze(4)  # (N, H, W, K, 9, 4)

    return torch.cat([splat_colors, splat_weights], dim=5)def _compute_splatted_colors_and_weights(
    occlusion_layers: torch.Tensor,  # (N, H, W, 9)
    splat_colors_and_weights: torch.Tensor,  # (N, H, W, K, 9, 5)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate splatted colors in background, surface and foreground occlusion buffers.

    Args:
        occlusion_layers: (N, H, W, 9) tensor. See _compute_occlusion_layers.
        splat_colors_and_weights: (N, H, W, K, 9, 5) tensor. See _offset_splats.

    Returns:
        splatted_colors: (N, H, W, 4, 3) tensor. Last dimension corresponds to back-
            ground, surface, and foreground splat colors.
        splatted_weights: (N, H, W, 1, 3) tensor. Last dimension corresponds to back-
            ground, surface, and foreground splat weights and is used for normalization.

    """
    N, H, W, K, _, _ = splat_colors_and_weights.shape

    # Create an occlusion mask, with the last dimension of length 3, corresponding to
    # background/surface/foreground splatting. E.g. occlusion_layer_mask[n,h,w,k,d,0] is
    # 1 if the pixel at hw is splatted from direction d such that the splatting pixel p
    # is below the splatted pixel q (in the background); otherwise, the value is 0.
    # occlusion_layer_mask[n,h,w,k,d,1] is 1 if the splatting pixel is at the same
    # surface level as the splatted pixel q, and occlusion_layer_mask[n,h,w,k,d,2] is
    # 1 only if the splatting pixel is in the foreground.
    layer_ids = torch.arange(K, device=splat_colors_and_weights.device).view(
        1, 1, 1, K, 1
    )
    occlusion_layers = occlusion_layers.view(N, H, W, 1, 9)
    occlusion_layer_mask = torch.stack(
        [
            occlusion_layers > layer_ids,  # (N, H, W, K, 9)
            occlusion_layers == layer_ids,  # (N, H, W, K, 9)
            occlusion_layers < layer_ids,  # (N, H, W, K, 9)
        ],
        dim=5,
    ).float()  # (N, H, W, K, 9, 3)

    # (N * H * W, 5, 9 * K) x (N * H * W, 9 * K, 3) -> (N * H * W, 5, 3)
    splatted_colors_and_weights = torch.bmm(
        splat_colors_and_weights.permute(0, 1, 2, 5, 3, 4).reshape(
            (N * H * W, 5, K * 9)
        ),
        occlusion_layer_mask.reshape((N * H * W, K * 9, 3)),
    ).reshape((N, H, W, 5, 3))

    return (
        splatted_colors_and_weights[..., :4, :],
        splatted_colors_and_weights[..., 4:5, :],
    )def _normalize_and_compose_all_layers(
    background_color: torch.Tensor,
    splatted_colors_per_occlusion_layer: torch.Tensor,
    splatted_weights_per_occlusion_layer: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize each bg/surface/fg buffer by its weight, and compose.

    Args:
        background_color: (3) RGB tensor.
        splatter_colors_per_occlusion_layer: (N, H, W, 4, 3) RGBA tensor, last dimension
            corresponds to foreground, surface, and background splatting.
        splatted_weights_per_occlusion_layer: (N, H, W, 1, 3) weight tensor.

    Returns:
        output_colors: (N, H, W, 4) RGBA tensor.
    """
    device = splatted_colors_per_occlusion_layer.device

    # Normalize each of bg/surface/fg splat layers separately.
    normalization_scales = 1.0 / (
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        torch.maximum(
            splatted_weights_per_occlusion_layer,
            torch.tensor([1.0], device=device),
        )
    )  # (N, H, W, 1, 3)

    normalized_splatted_colors = (
        splatted_colors_per_occlusion_layer * normalization_scales
    )  # (N, H, W, 4, 3)

    # Use alpha-compositing to compose the splat layers.
    output_colors = torch.cat(
        [background_color, torch.tensor([0.0], device=device)]
    )  # (4), will broadcast to (N, H, W, 4) below.

    for occlusion_layer_id in (-1, -2, -3):
        # Over-compose the bg, surface, and fg occlusion layers. Note that we already
        # multiplied each pixel's RGBA by its own alpha as part of self-splatting in
        # _compute_splatting_colors_and_weights, so we don't re-multiply by alpha here.
        alpha = normalized_splatted_colors[..., 3:4, occlusion_layer_id]  # (N, H, W, 1)
        output_colors = (
            normalized_splatted_colors[..., occlusion_layer_id]
            + (1.0 - alpha) * output_colors
        )
    return output_colors