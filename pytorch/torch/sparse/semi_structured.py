def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
    transposed: bool = False,
) -> SparseSemiStructuredTensor:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of the mininum sparse block size, given in
    `_DTYPE_TO_SHAPE_CONSTRAINTS` for each dtype (float32, float16, bfloat16, int8).

    Args:
        original_tensor (Tensor): the dense tensor to convert
        transposed (bool, optional): deprecated arg to be removed in another release. Do not use.
    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor
    Raises:
        None
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A)
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]))
        >>> A_sparse.values()
        tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
        >>> A_sparse.indices()
        tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0', dtype=torch.int16))
    """
    if transposed:
        warnings.warn(
            "Setting transpose from `to_sparse_semi_structured` is deprecated "
            "and will be removed in a future release. "
            "`SparseSemiStructuredTensor` only support contiguous input tensors.",
            FutureWarning,
            stacklevel=2,
        )

    # set from _FORCE_CUTLASS flag
    SPARSE_SUBCLASS = (
        torch.sparse.SparseSemiStructuredTensorCUTLASS
        if SparseSemiStructuredTensor._FORCE_CUTLASS
        else torch.sparse.SparseSemiStructuredTensorCUSPARSELT
    )

    return SPARSE_SUBCLASS.from_dense(original_tensor)