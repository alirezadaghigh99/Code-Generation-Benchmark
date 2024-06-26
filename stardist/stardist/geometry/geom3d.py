def polyhedron_to_label(dist, points, rays, shape, prob=None, thr=-np.inf, labels=None, mode="full", verbose=True, overlap_label=None):
    """
    creates labeled image from stardist representations

    :param dist: array of shape (n_points,n_rays)
        the list of distances for each point and ray
    :param points: array of shape (n_points, 3)
        the list of center points
    :param rays: Rays object
        Ray object (e.g. `stardist.Rays_GoldenSpiral`) defining
        vertices and faces
    :param shape: (nz,ny,nx)
        output shape of the image
    :param prob: array of length/shape (n_points,) or None
        probability per polyhedron
    :param thr: scalar
        probability threshold (only polyhedra with prob>thr are labeled)
    :param labels: array of length/shape (n_points,) or None
        labels to use
    :param mode: str
        labeling mode, can be "full", "kernel", "hull", "bbox" or  "debug"
    :param verbose: bool
        enable to print some debug messages
    :param overlap_label: scalar or None
        if given, will label each pixel that belongs ot more than one polyhedron with that label
    :return: array of given shape
        labeled image
    """
    if len(points) == 0:
        if verbose:
            print("warning: empty list of points (returning background-only image)")
        return np.zeros(shape, np.uint16)

    dist = np.asanyarray(dist)
    points = np.asanyarray(points)

    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if labels is None:
        labels = np.arange(1, len(points) + 1)

    if np.amin(dist) <= 0:
        raise ValueError("distance array should be positive!")

    prob = np.ones(len(points)) if prob is None else np.asanyarray(prob)

    if dist.ndim != 2:
        raise ValueError("dist should be 2 dimensional but has shape %s" % str(dist.shape))

    if dist.shape[1] != len(rays):
        raise ValueError("inconsistent number of rays!")

    if len(prob) != len(points):
        raise ValueError("len(prob) != len(points)")

    if len(labels) != len(points):
        raise ValueError("len(labels) != len(points)")

    modes = {"full": 0, "kernel": 1, "hull": 2, "bbox": 3, "debug": 4}

    if not mode in modes:
        raise KeyError("Unknown render mode '%s' , allowed:  %s" % (mode, tuple(modes.keys())))

    lbl = np.zeros(shape, np.uint16)

    # filter points
    ind = np.where(prob >= thr)[0]
    if len(ind) == 0:
        if verbose:
            print("warning: no points found with probability>= {thr:.4f} (returning background-only image)".format(thr=thr))
        return lbl

    prob = prob[ind]
    points = points[ind]
    dist = dist[ind]
    labels = labels[ind]

    # sort points with decreasing probability
    ind = np.argsort(prob)[::-1]
    points = points[ind]
    dist = dist[ind]
    labels = labels[ind]

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    return c_polyhedron_to_label(_prep(dist, np.float32),
                                 _prep(points, np.float32),
                                 _prep(rays.vertices, np.float32),
                                 _prep(rays.faces, np.int32),
                                 _prep(labels, np.int32),
                                 np.int32(modes[mode]),
                                 np.int32(verbose),
                                 np.int32(overlap_label is not None),
                                 np.int32(0 if overlap_label is None else overlap_label),
                                 shape
                                 )