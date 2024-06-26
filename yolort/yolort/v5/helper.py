def get_yolov5_size(depth_multiple, width_multiple):
    if depth_multiple == 0.33 and width_multiple == 0.25:
        return "n"
    if depth_multiple == 0.33 and width_multiple == 0.5:
        return "s"
    if depth_multiple == 0.67 and width_multiple == 0.75:
        return "m"
    if depth_multiple == 1.0 and width_multiple == 1.0:
        return "l"
    if depth_multiple == 1.33 and width_multiple == 1.25:
        return "x"
    raise NotImplementedError(
        f"Currently does't support architecture with depth: {depth_multiple} "
        f"and {width_multiple}, fell free to create a ticket labeled enhancement to us"
    )