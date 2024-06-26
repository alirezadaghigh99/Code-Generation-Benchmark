def parse_polygon_points(polygon: Element) -> np.ndarray:
    coordinates = [int(coord.text) for coord in polygon.findall(".//*")]
    return np.array(
        [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]
    )