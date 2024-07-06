def get_face_boxes(coords: np.ndarray, pad: float = 5.0) -> List[CoordinateBox]:
    """For each face of the convex hull, compute a coordinate box around it.

    The convex hull of a macromolecule will have a series of triangular
    faces. For each such triangular face, we construct a bounding box
    around this triangle. Think of this box as attempting to capture
    some binding interaction region whose exterior is controlled by the
    box. Note that this box will likely be a crude approximation, but
    the advantage of this technique is that it only uses simple geometry
    to provide some basic biological insight into the molecule at hand.

    The `pad` parameter is used to control the amount of padding around
    the face to be used for the coordinate box.

    Parameters
    ----------
    coords: np.ndarray
        A numpy array of shape `(N, 3)`. The coordinates of a molecule.
    pad: float, optional (default 5.0)
        The number of angstroms to pad.

    Returns
    -------
    boxes: List[CoordinateBox]
        List of `CoordinateBox`

    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> boxes = get_face_boxes(coords, pad=5)
    """
    hull = ConvexHull(coords)
    boxes = []
    # Each triangle in the simplices is a set of 3 atoms from
    # coordinates which forms the vertices of an exterior triangle on
    # the convex hull of the macromolecule.
    for triangle in hull.simplices:
        # Points is the set of atom coordinates that make up this
        # triangular face on the convex hull
        points = np.array(
            [coords[triangle[0]], coords[triangle[1]], coords[triangle[2]]])
        # Let's extract x/y/z coords for this face
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]

        # Let's compute min/max points
        x_min, x_max = np.amin(x_coords), np.amax(x_coords)
        x_min, x_max = int(np.floor(x_min)) - pad, int(np.ceil(x_max)) + pad
        x_bounds = (x_min, x_max)

        y_min, y_max = np.amin(y_coords), np.amax(y_coords)
        y_min, y_max = int(np.floor(y_min)) - pad, int(np.ceil(y_max)) + pad
        y_bounds = (y_min, y_max)
        z_min, z_max = np.amin(z_coords), np.amax(z_coords)
        z_min, z_max = int(np.floor(z_min)) - pad, int(np.ceil(z_max)) + pad
        z_bounds = (z_min, z_max)
        box = CoordinateBox(x_bounds, y_bounds, z_bounds)
        boxes.append(box)
    return boxes

