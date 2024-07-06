def full(size, fill_value, *args, **kwargs):
    dtype = kwargs.get("dtype", None)
    if not dtype:
        dtype = utils.get_dtype(fill_value)
    kwargs["dtype"] = dtype
    return torch.empty(size, *args, **kwargs)

def zeros_like(
    self,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if layout == torch.sparse_coo:
        torch._check(
            memory_format is None,
            lambda: "memory format option is only supported by strided tensors",
        )

        res = torch.empty(
            0,
            dtype=self.dtype if dtype is None else dtype,
            layout=layout,
            device=self.device if device is None else device,
            pin_memory=pin_memory,
        )

        if self.is_sparse:
            res.sparse_resize_and_clear_(
                self.size(), self.sparse_dim(), self.dense_dim()
            )
        else:
            res.sparse_resize_and_clear_(self.size(), self.dim(), 0)

        res._coalesced_(True)
        return res
    res = aten.empty_like.default(
        self,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )
    # device can be not "meta"
    res.fill_(0)
    return res

def nan_to_num(self, nan=None, posinf=None, neginf=None):
    result_size = list(self.size())
    return self.new_empty(result_size)

def zeros_like(
    self,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if layout == torch.sparse_coo:
        torch._check(
            memory_format is None,
            lambda: "memory format option is only supported by strided tensors",
        )

        res = torch.empty(
            0,
            dtype=self.dtype if dtype is None else dtype,
            layout=layout,
            device=self.device if device is None else device,
            pin_memory=pin_memory,
        )

        if self.is_sparse:
            res.sparse_resize_and_clear_(
                self.size(), self.sparse_dim(), self.dense_dim()
            )
        else:
            res.sparse_resize_and_clear_(self.size(), self.dim(), 0)

        res._coalesced_(True)
        return res
    res = aten.empty_like.default(
        self,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )
    # device can be not "meta"
    res.fill_(0)
    return res

