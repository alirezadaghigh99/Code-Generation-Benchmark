def equalize_clahe(
    input: torch.Tensor,
    clip_limit: float = 40.0,
    grid_size: Tuple[int, int] = (8, 8),
    slow_and_differentiable: bool = False,
) -> torch.Tensor:
    r"""Apply clahe equalization on the input tensor.

    .. image:: _static/img/equalize_clahe.png

    NOTE: Lut computation uses the same approach as in OpenCV, in next versions this can change.

    Args:
        input: images tensor to equalize with values in the range [0, 1] and shape :math:`(*, C, H, W)`.
        clip_limit: threshold value for contrast limiting. If 0 clipping is disabled.
        grid_size: number of tiles to be cropped in each direction (GH, GW).
        slow_and_differentiable: flag to select implementation

    Returns:
        Equalized image or images with shape as the input.

    Examples:
        >>> img = torch.rand(1, 10, 20)
        >>> res = equalize_clahe(img)
        >>> res.shape
        torch.Size([1, 10, 20])

        >>> img = torch.rand(2, 3, 10, 20)
        >>> res = equalize_clahe(img)
        >>> res.shape
        torch.Size([2, 3, 10, 20])
    """
    if not isinstance(clip_limit, float):
        raise TypeError(f"Input clip_limit type is not float. Got {type(clip_limit)}")

    if not isinstance(grid_size, tuple):
        raise TypeError(f"Input grid_size type is not Tuple. Got {type(grid_size)}")

    if len(grid_size) != 2:
        raise TypeError(f"Input grid_size is not a Tuple with 2 elements. Got {len(grid_size)}")

    if isinstance(grid_size[0], float) or isinstance(grid_size[1], float):
        raise TypeError("Input grid_size type is not valid, must be a Tuple[int, int].")

    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError(f"Input grid_size elements must be positive. Got {grid_size}")

    imgs: torch.Tensor = input  # B x C x H x W

    # hist_tiles: torch.Tensor  # B x GH x GW x C x TH x TW  # not supported by JIT
    # img_padded: torch.Tensor  # B x C x H' x W'  # not supported by JIT
    # the size of the tiles must be even in order to divide them into 4 tiles for the interpolation
    hist_tiles, img_padded = _compute_tiles(imgs, grid_size, True)
    tile_size: Tuple[int, int] = (hist_tiles.shape[-2], hist_tiles.shape[-1])
    interp_tiles: torch.Tensor = _compute_interpolation_tiles(img_padded, tile_size)  # B x 2GH x 2GW x C x TH/2 x TW/2
    luts: torch.Tensor = _compute_luts(
        hist_tiles, clip=clip_limit, diff=slow_and_differentiable
    )  # B x GH x GW x C x 256
    equalized_tiles: torch.Tensor = _compute_equalized_tiles(interp_tiles, luts)  # B x 2GH x 2GW x C x TH/2 x TW/2

    # reconstruct the images form the tiles
    #    try permute + contiguous + view
    eq_imgs: torch.Tensor = equalized_tiles.permute(0, 3, 1, 4, 2, 5).reshape_as(img_padded)
    h, w = imgs.shape[-2:]
    eq_imgs = eq_imgs[..., :h, :w]  # crop imgs if they were padded

    # remove batch if the input was not in batch form
    if input.dim() != eq_imgs.dim():
        eq_imgs = eq_imgs.squeeze(0)

    return eq_imgs

