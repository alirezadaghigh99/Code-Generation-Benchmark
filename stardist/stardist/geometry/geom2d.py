def polygons_to_label_coord(coord, shape, labels=None):
    """renders polygons to image of given shape

    coord.shape   = (n_polys, n_rays)
    """
    coord = np.asarray(coord)
    if labels is None: labels = np.arange(len(coord))

    _check_label_array(labels, "labels")
    assert coord.ndim==3 and coord.shape[1]==2 and len(coord)==len(labels)

    lbl = np.zeros(shape,np.int32)

    for i,c in zip(labels,coord):
        rr,cc = polygon(*c, shape)
        lbl[rr,cc] = i+1

    return lbl