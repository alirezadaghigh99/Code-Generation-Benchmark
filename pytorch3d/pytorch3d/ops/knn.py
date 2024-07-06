def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
) -> _KNN:
    """
    K-Nearest neighbors on point clouds.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
            containing up to P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
            containing up to P2 points of dimension D.
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
            length of each pointcloud in p1. Or None to indicate that every cloud has
            length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
            length of each pointcloud in p2. Or None to indicate that every cloud has
            length P2.
        norm: Integer indicating the norm of the distance. Supports only 1 for L1, 2 for L2.
        K: Integer giving the number of nearest neighbors to return.
        version: Which KNN implementation to use in the backend. If version=-1,
            the correct implementation is selected based on the shapes of the inputs.
        return_nn: If set to True returns the K nearest neighbors in p2 for each point in p1.
        return_sorted: (bool) whether to return the nearest neighbors sorted in
            ascending order of distance.

    Returns:
        dists: Tensor of shape (N, P1, K) giving the squared distances to
            the nearest neighbors. This is padded with zeros both where a cloud in p2
            has fewer than K points and where a cloud in p1 has fewer than P1 points.

        idx: LongTensor of shape (N, P1, K) giving the indices of the
            K nearest neighbors from points in p1 to points in p2.
            Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
            neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
            in p2 has fewer than K points and where a cloud in p1 has fewer than P1
            points.

        nn: Tensor of shape (N, P1, K, D) giving the K nearest neighbors in p2 for
            each point in p1. Concretely, `p2_nn[n, i, k]` gives the k-th nearest neighbor
            for `p1[n, i]`. Returned if `return_nn` is True.
            The nearest neighbors are collected using `knn_gather`

            .. code-block::

                p2_nn = knn_gather(p2, p1_idx, lengths2)

            which is a helper function that allows indexing any tensor of shape (N, P2, U) with
            the indices `p1_idx` returned by `knn_points`. The output is a tensor
            of shape (N, P1, K, U).

    """
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("pts1 and pts2 must have the same point dimension.")

    p1 = p1.contiguous()
    p2 = p2.contiguous()

    P1 = p1.shape[1]
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((p1.shape[0],), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((p1.shape[0],), P2, dtype=torch.int64, device=p1.device)

    p1_dists, p1_idx = _knn_points.apply(
        p1, p2, lengths1, lengths2, K, version, norm, return_sorted
    )

    p2_nn = None
    if return_nn:
        p2_nn = knn_gather(p2, p1_idx, lengths2)

    return _KNN(dists=p1_dists, idx=p1_idx, knn=p2_nn if return_nn else None)

