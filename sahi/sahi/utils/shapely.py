def get_shapely_box(x: int, y: int, width: int, height: int) -> Polygon:
    """
    Accepts coco style bbox coords and converts it to shapely box object
    """
    minx = x
    miny = y
    maxx = x + width
    maxy = y + height
    shapely_box = box(minx, miny, maxx, maxy)

    return shapely_box

