def merge_points(points, radius):
    """
    Greedily merge points that are closer than given radius.

    This uses :class:`LSH` to achieve complexity that is linear in the number
    of merged clusters and quadratic in the size of the largest merged cluster.

    :param torch.Tensor points: A tensor of shape ``(K,D)`` where ``K`` is
        the number of points and ``D`` is the number of dimensions.
    :param float radius: The minimum distance nearer than which
        points will be merged.
    :return: A tuple ``(merged_points, groups)`` where ``merged_points`` is a
        tensor of shape ``(J,D)`` where ``J <= K``, and ``groups`` is a list of
        tuples of indices mapping merged points to original points. Note that
        ``len(groups) == J`` and ``sum(len(group) for group in groups) == K``.
    :rtype: tuple
    """
    if points.dim() != 2:
        raise ValueError(
            "Expected points.shape == (K,D), but got {}".format(points.shape)
        )
    if not (isinstance(radius, Number) and radius > 0):
        raise ValueError(
            "Expected radius to be a positive number, but got {}".format(radius)
        )
    radius = (
        0.99 * radius
    )  # avoid merging points exactly radius apart, e.g. grid points
    threshold = radius**2

    # setup data structures to cheaply search for nearest pairs
    lsh = LSH(radius)
    priority_queue = []
    groups = [(i,) for i in range(len(points))]
    for i, point in enumerate(points):
        lsh.add(i, point)
        for j in lsh.nearby(i):
            d2 = (point - points[j]).pow(2).sum().item()
            if d2 < threshold:
                heapq.heappush(priority_queue, (d2, j, i))
    if not priority_queue:
        return points, groups

    # convert from dense to sparse representation
    next_id = len(points)
    points = dict(enumerate(points))
    groups = dict(enumerate(groups))

    # greedily merge
    while priority_queue:
        d1, i, j = heapq.heappop(priority_queue)
        if i not in points or j not in points:
            continue
        k = next_id
        next_id += 1
        points[k] = (points.pop(i) + points.pop(j)) / 2
        groups[k] = groups.pop(i) + groups.pop(j)
        lsh.remove(i)
        lsh.remove(j)
        lsh.add(k, points[k])
        for i in lsh.nearby(k):
            if i == k:
                continue
            d2 = (points[i] - points[k]).pow(2).sum().item()
            if d2 < threshold:
                heapq.heappush(priority_queue, (d2, i, k))

    # convert from sparse to dense representation
    ids = sorted(points.keys())
    points = torch.stack([points[i] for i in ids])
    groups = [groups[i] for i in ids]

    return points, groups

