def star_dist(a, n_rays=32, grid=(1,1), mode='cpp'):
    """'a' assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""

    n_rays >= 3 or _raise(ValueError("need 'n_rays' >= 3"))

    if mode == 'python':
        return _py_star_dist(a, n_rays, grid=grid)
    elif mode == 'cpp':
        return _cpp_star_dist(a, n_rays, grid=grid)
    elif mode == 'opencl':
        return _ocl_star_dist(a, n_rays, grid=grid)
    else:
        _raise(ValueError("Unknown mode %s" % mode))