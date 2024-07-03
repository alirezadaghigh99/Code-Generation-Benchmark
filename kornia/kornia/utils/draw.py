def draw_line(image: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    r"""Draw a single line into an image.

    Args:
        image: the input image to where to draw the lines with shape :math`(C,H,W)`.
        p1: the start point [x y] of the line with shape (2, ) or (B, 2).
        p2: the end point [x y] of the line with shape (2, ) or (B, 2).
        color: the color of the line with shape :math`(C)` where :math`C` is the number of channels of the image.
    Return:
        the image with containing the line.

    Examples:
        >>> image = torch.zeros(1, 8, 8)
        >>> draw_line(image, torch.tensor([6, 4]), torch.tensor([1, 4]), torch.tensor([255]))
        tensor([[[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0., 255., 255., 255., 255., 255., 255.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]])
    """
    if (p1.shape[0] != p2.shape[0]) or (p1.shape[-1] != 2 or p2.shape[-1] != 2):
        raise ValueError(
            "Input points must be 2D points with shape (2, ) or (B, 2) and must have the same batch sizes."
        )
    if (
        (p1[..., 0] < 0).any()
        or (p1[..., 0] >= image.shape[-1]).any()
        or (p1[..., 1] < 0).any()
        or (p1[..., 1] >= image.shape[-2]).any()
    ):
        raise ValueError("p1 is out of bounds.")
    if (
        (p2[..., 0] < 0).any()
        or (p2[..., 0] >= image.shape[-1]).any()
        or (p2[..., 1] < 0).any()
        or (p2[..., 1] >= image.shape[-2]).any()
    ):
        raise ValueError("p2 is out of bounds.")

    if len(image.size()) != 3:
        raise ValueError("image must have 3 dimensions (C,H,W).")

    if color.size(0) != image.size(0):
        raise ValueError("color must have the same number of channels as the image.")

    # move p1 and p2 to the same device as the input image
    # move color to the same device and dtype as the input image
    p1 = p1.to(image.device).to(torch.int64)
    p2 = p2.to(image.device).to(torch.int64)
    color = color.to(image)

    x1, y1 = p1[..., 0], p1[..., 1]
    x2, y2 = p2[..., 0], p2[..., 1]
    dx = x2 - x1
    dy = y2 - y1
    dx_sign = torch.sign(dx)
    dy_sign = torch.sign(dy)
    dx, dy = torch.abs(dx), torch.abs(dy)
    dx_zero_mask = dx == 0
    dy_zero_mask = dy == 0
    dx_gt_dy_mask = (dx > dy) & ~(dx_zero_mask | dy_zero_mask)
    rest_mask = ~(dx_zero_mask | dy_zero_mask | dx_gt_dy_mask)

    dx_zero_x_coords, dx_zero_y_coords = [], []
    dy_zero_x_coords, dy_zero_y_coords = [], []
    dx_gt_dy_x_coords, dx_gt_dy_y_coords = [], []
    rest_x_coords, rest_y_coords = [], []

    if dx_zero_mask.any():
        dx_zero_x_coords = [
            x for x_i, dy_i in zip(x1[dx_zero_mask], dy[dx_zero_mask]) for x in x_i.repeat(int(dy_i.item() + 1))
        ]
        dx_zero_y_coords = [
            y
            for y_i, s, dy_ in zip(y1[dx_zero_mask], dy_sign[dx_zero_mask], dy[dx_zero_mask])
            for y in (y_i + s * torch.arange(0, dy_ + 1, 1, device=image.device))
        ]

    if dy_zero_mask.any():
        dy_zero_x_coords = [
            x
            for x_i, s, dx_i in zip(x1[dy_zero_mask], dx_sign[dy_zero_mask], dx[dy_zero_mask])
            for x in (x_i + s * torch.arange(0, dx_i + 1, 1, device=image.device))
        ]
        dy_zero_y_coords = [
            y for y_i, dx_i in zip(y1[dy_zero_mask], dx[dy_zero_mask]) for y in y_i.repeat(int(dx_i.item() + 1))
        ]

    if dx_gt_dy_mask.any():
        dx_gt_dy_x_coords = [
            x
            for x_i, s, dx_i in zip(x1[dx_gt_dy_mask], dx_sign[dx_gt_dy_mask], dx[dx_gt_dy_mask])
            for x in (x_i + s * torch.arange(0, dx_i + 1, 1, device=image.device))
        ]
        dx_gt_dy_y_coords = [
            y
            for y_i, s, dx_i, dy_i in zip(
                y1[dx_gt_dy_mask], dy_sign[dx_gt_dy_mask], dx[dx_gt_dy_mask], dy[dx_gt_dy_mask]
            )
            for y in (
                y_i + s * torch.arange(0, dy_i + 1, dy_i / dx_i, device=image.device)[: int(dx_i.item()) + 1].ceil()
            )
        ]
    if rest_mask.any():
        rest_x_coords = [
            x
            for x_i, s, dx_i, dy_ in zip(x1[rest_mask], dx_sign[rest_mask], dx[rest_mask], dy[rest_mask])
            for x in (
                x_i + s * torch.arange(0, dx_i + 1, dx_i / dy_, device=image.device)[: int(dy_.item()) + 1].ceil()
            )
        ]
        rest_y_coords = [
            y
            for y_i, s, dy_i in zip(y1[rest_mask], dy_sign[rest_mask], dy[rest_mask])
            for y in (y_i + s * torch.arange(0, dy_i + 1, 1, device=image.device))
        ]
    x_coords = torch.tensor(dx_zero_x_coords + dy_zero_x_coords + dx_gt_dy_x_coords + rest_x_coords).long()
    y_coords = torch.tensor(dx_zero_y_coords + dy_zero_y_coords + dx_gt_dy_y_coords + rest_y_coords).long()
    image[:, y_coords, x_coords] = color.view(-1, 1)
    return imagedef draw_rectangle(
    image: torch.Tensor, rectangle: torch.Tensor, color: Optional[torch.Tensor] = None, fill: Optional[bool] = None
) -> torch.Tensor:
    r"""Draw N rectangles on a batch of image tensors.

    Args:
        image: is tensor of BxCxHxW.
        rectangle: represents number of rectangles to draw in BxNx4
            N is the number of boxes to draw per batch index[x1, y1, x2, y2]
            4 is in (top_left.x, top_left.y, bot_right.x, bot_right.y).
        color: a size 1, size 3, BxNx1, or BxNx3 tensor.
            If C is 3, and color is 1 channel it will be broadcasted.
        fill: is a flag used to fill the boxes with color if True.

    Returns:
        This operation modifies image inplace but also returns the drawn tensor for
        convenience with same shape the of the input BxCxHxW.

    Example:
        >>> img = torch.rand(2, 3, 10, 12)
        >>> rect = torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]]])
        >>> out = draw_rectangle(img, rect)
    """

    batch, c, h, w = image.shape
    batch_rect, num_rectangle, num_points = rectangle.shape
    if batch != batch_rect:
        raise AssertionError("Image batch and rectangle batch must be equal")
    if num_points != 4:
        raise AssertionError("Number of points in rectangle must be 4")

    # clone rectangle, in case it's been expanded assignment from clipping causes problems
    rectangle = rectangle.long().clone()

    # clip rectangle to hxw bounds
    rectangle[:, :, 1::2] = torch.clamp(rectangle[:, :, 1::2], 0, h - 1)
    rectangle[:, :, ::2] = torch.clamp(rectangle[:, :, ::2], 0, w - 1)

    if color is None:
        color = torch.tensor([0.0] * c).expand(batch, num_rectangle, c)

    if fill is None:
        fill = False

    if len(color.shape) == 1:
        color = color.expand(batch, num_rectangle, c)
    b, n, color_channels = color.shape

    if color_channels == 1 and c == 3:
        color = color.expand(batch, num_rectangle, c)

    for b in range(batch):
        for n in range(num_rectangle):
            if fill:
                image[
                    b,
                    :,
                    int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1),
                    int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1),
                ] = color[b, n, :, None, None]
            else:
                image[b, :, int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1), rectangle[b, n, 0]] = color[
                    b, n, :, None
                ]
                image[b, :, int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1), rectangle[b, n, 2]] = color[
                    b, n, :, None
                ]
                image[b, :, rectangle[b, n, 1], int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1)] = color[
                    b, n, :, None
                ]
                image[b, :, rectangle[b, n, 3], int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1)] = color[
                    b, n, :, None
                ]

    return image