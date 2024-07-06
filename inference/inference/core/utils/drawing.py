def create_tiles(
    images: List[np.ndarray],
    grid_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
    single_tile_size: Optional[Tuple[int, int]] = None,
    tile_scaling: Literal["min", "max", "avg"] = "avg",
    tile_padding_color: Tuple[int, int, int] = (0, 0, 0),
    tile_margin: int = 15,
    tile_margin_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("Could not create image tiles from empty list of images.")
    if single_tile_size is None:
        single_tile_size = _aggregate_images_shape(images=images, mode=tile_scaling)
    resized_images = [
        letterbox_image(
            image=i, desired_size=single_tile_size, color=tile_padding_color
        )
        for i in images
    ]
    grid_size = _establish_grid_size(images=images, grid_size=grid_size)
    if len(images) > grid_size[0] * grid_size[1]:
        raise ValueError(f"Grid of size: {grid_size} cannot fit {len(images)} images.")
    return _generate_tiles(
        images=resized_images,
        grid_size=grid_size,
        single_tile_size=single_tile_size,
        tile_padding_color=tile_padding_color,
        tile_margin=tile_margin,
        tile_margin_color=tile_margin_color,
    )

