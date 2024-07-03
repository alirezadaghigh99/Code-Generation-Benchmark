def get_box_from_mask(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Find if there are non-zero elements per row/column first and then find
    min/max position of those elements.
    Only support 2d image (h x w)
    Return (x1, y1, w, h) if bbox found, otherwise None
    """
    assert len(mask.shape) == 2, f"Invalid shape {mask.shape}"
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if bool(np.any(rows)) is False or bool(np.any(cols)) is False:
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    assert cmax >= cmin, f"cmax={cmax}, cmin={cmin}"
    assert rmax >= rmin, f"rmax={rmax}, rmin={rmin}"

    # x1, y1, w, h
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1