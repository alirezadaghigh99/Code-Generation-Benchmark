def calculate_extents(lbl, func=np.median):
    """ Aggregate bounding box sizes of objects in label images. """
    if (isinstance(lbl,np.ndarray) and lbl.ndim==4) or (not isinstance(lbl,np.ndarray) and  isinstance(lbl,Iterable)):
        return func(np.stack([calculate_extents(_lbl,func) for _lbl in lbl], axis=0), axis=0)

    n = lbl.ndim
    n in (2,3) or _raise(ValueError("label image should be 2- or 3-dimensional (or pass a list of these)"))

    regs = regionprops(lbl)
    if len(regs) == 0:
        return np.zeros(n)
    else:
        extents = np.array([np.array(r.bbox[n:])-np.array(r.bbox[:n]) for r in regs])
        return func(extents, axis=0)

