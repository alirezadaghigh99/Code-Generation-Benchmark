def non_maximum_suppression_3d(dist, prob, rays, grid=(1,1,1), b=2, nms_thresh=0.5, prob_thresh=0.5, use_bbox=True, use_kdtree=True, verbose=False):
    """Non-Maximum-Supression of 3D polyhedra

    Retains only polyhedra whose overlap is smaller than nms_thresh

    dist.shape = (Nz,Ny,Nx, n_rays)
    prob.shape = (Nz,Ny,Nx)

    returns the retained points, probabilities, and distances:

    points, prob, dist = non_maximum_suppression_3d(dist, prob, ....
    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)

    assert prob.ndim == 3 and dist.ndim == 4 and dist.shape[-1] == len(rays) and prob.shape == dist.shape[:3]

    grid = _normalize_grid(grid,3)

    verbose and print("predicting instances with prob_thresh = {prob_thresh} and nms_thresh = {nms_thresh}".format(prob_thresh=prob_thresh, nms_thresh=nms_thresh), flush=True)

    # ind_thresh = prob > prob_thresh
    # if b is not None and b > 0:
    #     _ind_thresh = np.zeros_like(ind_thresh)
    #     _ind_thresh[b:-b,b:-b,b:-b] = True
    #     ind_thresh &= _ind_thresh

    ind_thresh = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(ind_thresh), axis=1)
    verbose and print("found %s candidates"%len(points))
    probi = prob[ind_thresh]
    disti = dist[ind_thresh]

    _sorted = np.argsort(probi)[::-1]
    probi = probi[_sorted]
    disti = disti[_sorted]
    points = points[_sorted]

    verbose and print("non-maximum suppression...")
    points = (points * np.array(grid).reshape((1,3)))

    inds = non_maximum_suppression_3d_inds(disti, points, rays=rays, scores=probi, thresh=nms_thresh,
                                           use_bbox=use_bbox, use_kdtree = use_kdtree,
                                           verbose=verbose)

    verbose and print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
    return points[inds], probi[inds], disti[inds]