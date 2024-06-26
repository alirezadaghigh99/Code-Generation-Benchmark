def _generate_color_image(
    shape: Tuple[int, int], color: Tuple[int, int, int]
) -> np.ndarray:
    return np.ones(shape[::-1] + (3,), dtype=np.uint8) * color