def sample_farthest_points_naive(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same Args/Returns as sample_farthest_points
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("Invalid lengths.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_K = torch.max(K)

    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(N):
        # Initialize an array for the sampled indices, shape: (max_K,)
        sample_idx_batch = torch.full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (max_K,),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        closest_dists = points.new_full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (lengths[n],),
            float("inf"),
            dtype=torch.float32,
        )

        # Select a random point index and save it as the starting point
        # pyre-fixme[6]: For 2nd argument expected `int` but got `Tensor`.
        selected_idx = randint(0, lengths[n] - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        k_n = min(lengths[n], K[n])

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # it's distance will be 0.0 so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            selected_idx = torch.argmax(closest_dists)
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

    # Gather the points
    all_sampled_points = masked_gather(points, all_sampled_indices)

    # Return (N, max_K, D) subsampled points and indices
    return all_sampled_points, all_sampled_indices

