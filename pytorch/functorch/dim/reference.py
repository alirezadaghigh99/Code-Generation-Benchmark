def stack(tensors, new_dim, dim=0, out=None):
    if isinstance(dim, int):
        return torch.stack(tensors, dim, out).index(dim, new_dim)
    index = None
    if out is not None:
        out, index = _positional_no_permute(out, dim, expand_dim=True)
    ptensors = []
    for t in tensors:
        pt, pi = _positional_no_permute(t, dim, expand_dim=True)
        if index is not None and pi != index:
            pt = pt.move_dim(pi, index)
        else:
            index = pi
        ptensors.append(pt)
    pr = torch.stack(ptensors, index, out=out)
    return pr.index((index, index + 1), (new_dim, dim))def split(self, split_size_or_sections, dim=0):
    from . import _Tensor, Dim

    if isinstance(split_size_or_sections, int) or any(
        isinstance(t, int) for t in split_size_or_sections
    ):
        if isinstance(dim, Dim):
            raise ValueError(
                "when dim is specified as a Dim object, split sizes must also be dimensions."
            )
        return _orig_split(self, split_size_or_sections, dim=dim)

    if isinstance(dim, Dim):
        assert isinstance(self, _Tensor), f"Tensor does not have dimension {dim}"
        self, dim = _positional_no_permute(self, dim)

    size = self.size(dim)
    total_bound_size = 0
    unbound = []
    sizes = []
    for i, d in enumerate(split_size_or_sections):
        if d.is_bound:
            sizes.append(d.size)
            total_bound_size += d.size
        else:
            sizes.append(0)
            unbound.append(i)

    if unbound:
        assert (
            total_bound_size <= size
        ), f"result dimensions are larger than original: {total_bound_size} vs {size} ({split_size_or_sections})"
        remaining_size = size - total_bound_size
        chunk_size = -(-remaining_size // len(unbound))
        for u in unbound:
            sz = min(chunk_size, remaining_size)
            split_size_or_sections[u].size = sz
            sizes[u] = sz
            remaining_size -= sz
    else:
        assert (
            total_bound_size == size
        ), f"result dimensions do not match original: {total_bound_size} vs {size} ({split_size_or_sections})"
    return tuple(
        t.index(dim, d)
        for d, t in zip(split_size_or_sections, _orig_split(self, sizes, dim=dim))
    )