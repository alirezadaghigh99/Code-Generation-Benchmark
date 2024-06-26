def expand_line(line: np.ndarray, target_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Expands a 2-point line, so that the first is on the edge. In other terms, we extend the line in
    the same direction until we meet one of the edges.

    Args:
    ----
        line: array of shape (2, 2) of the point supposed to be on one edge, and the shadow tip.
        target_shape: the desired mask shape

    Returns:
    -------
        2D coordinates of the first point once we extended the line (on one of the edges)
    """
    if any(coord == 0 or coord == size for coord, size in zip(line[0], target_shape[::-1])):
        return line[0]
    # Get the line equation
    _tmp = line[1] - line[0]
    _direction = _tmp > 0
    _flat = _tmp == 0
    # vertical case
    if _tmp[0] == 0:
        solutions = [
            # y = 0
            (line[0, 0], 0),
            # y = bot
            (line[0, 0], target_shape[0]),
        ]
    # horizontal
    elif _tmp[1] == 0:
        solutions = [
            # x = 0
            (0, line[0, 1]),
            # x = right
            (target_shape[1], line[0, 1]),
        ]
    else:
        alpha = _tmp[1] / _tmp[0]
        beta = line[1, 1] - alpha * line[1, 0]

        # Solve it for edges
        solutions = [
            # x = 0
            (0, beta),
            # y = 0
            (-beta / alpha, 0),
            # x = right
            (target_shape[1], alpha * target_shape[1] + beta),
            # y = bot
            ((target_shape[0] - beta) / alpha, target_shape[0]),
        ]
    for point in solutions:
        # Skip points that are out of the final image
        if any(val < 0 or val > size for val, size in zip(point, target_shape[::-1])):
            continue
        if all(
            val == ref if _same else (val < ref if _dir else val > ref)
            for val, ref, _dir, _same in zip(point, line[1], _direction, _flat)
        ):
            return point
    raise ValueError