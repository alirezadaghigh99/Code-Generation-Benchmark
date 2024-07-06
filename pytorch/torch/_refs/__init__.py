def addcdiv(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.addcdiv
    """
    if value is not None:
        dtype = self.dtype  # no scalars allowed, see add
        python_type = utils.dtype_to_type(dtype)
        torch._check_value(
            utils.is_weakly_lesser_type(type(value), python_type),
            lambda: f"value argument of type {type(value)} cannot be safely cast to type {python_type}!",
        )

    return self + value * tensor1 / tensor2

def diag_embed(
    t: TensorLikeType,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    """
    Reference implementation of torch.diag_embed
    """
    # convert from negative dims
    rank = t.ndim + 1
    dim1 = utils.canonicalize_dim(rank=rank, idx=dim1)
    dim2 = utils.canonicalize_dim(rank=rank, idx=dim2)

    # as per the docs, exchanging dims is equivalent to changing the sign of
    # offset
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset

    torch._check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    # as per the docs, the size of last dim is placed at dim1 and dim2
    last_dim = t.size(-1)

    if offset != 0:
        # add padding to match the new size
        t_shape = list(t.shape)
        t_shape[-1] = builtins.abs(offset)
        z = torch.zeros(t_shape, dtype=t.dtype, device=t.device, requires_grad=False)
        pair = (z, t) if offset > 0 else (t, z)
        t = torch.cat(pair, dim=-1)
        # make sure the diagonal always has the same size
        last_dim += builtins.abs(offset)

    # preserve original data, but place 1 at dim1 and move last dim to dim2
    t = t.unsqueeze(dim1).movedim(-1, dim2)

    # generate ranges shifting indices based on offset
    a_range = torch.arange(last_dim, device=t.device, dtype=torch.int64)
    b_range = torch.arange(
        offset, last_dim + offset, device=t.device, dtype=torch.int64
    )

    # broadcast
    cond = a_range == b_range.unsqueeze(-1)
    cond_shape = [last_dim if i in (dim1, dim2) else 1 for i in range(len(t.shape))]
    cond = cond.reshape(cond_shape)

    # aten.diag_embed always returns a new contiguous tensor
    # contiguous() is needed to correctly model the output stride
    return utils.mask_tensor(cond, t).contiguous()

