def random_bbox_assignment(
    dataset: GeoDataset,
    lengths: Sequence[float],
    generator: Generator | None = default_generator,
) -> list[GeoDataset]:
    """Split a GeoDataset randomly assigning its index's BoundingBoxes.

    This function will go through each BoundingBox in the GeoDataset's index and
    randomly assign it to new GeoDatasets.

    Args:
        dataset: dataset to be split
        lengths: lengths or fractions of splits to be produced
        generator: (optional) generator used for the random permutation

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.5
    """
    if not (isclose(sum(lengths), 1) or isclose(sum(lengths), len(dataset))):
        raise ValueError(
            "Sum of input lengths must equal 1 or the length of dataset's index."
        )

    if any(n <= 0 for n in lengths):
        raise ValueError('All items in input lengths must be greater than 0.')

    if isclose(sum(lengths), 1):
        lengths = _fractions_to_lengths(lengths, len(dataset))
    lengths = cast(Sequence[int], lengths)

    hits = list(dataset.index.intersection(dataset.index.bounds, objects=True))

    hits = [hits[i] for i in randperm(sum(lengths), generator=generator)]

    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in lengths
    ]

    for i, length in enumerate(lengths):
        for j in range(length):
            hit = hits.pop()
            new_indexes[i].insert(j, hit.bounds, hit.object)

    new_datasets = []
    for index in new_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        new_datasets.append(ds)

    return new_datasets

def time_series_split(
    dataset: GeoDataset, lengths: Sequence[float | tuple[float, float]]
) -> list[GeoDataset]:
    """Split a GeoDataset on its time dimension to create non-overlapping GeoDatasets.

    Args:
        dataset: dataset to be split
        lengths: lengths, fractions or pairs of timestamps (start, end) of splits
            to be produced

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.5
    """
    minx, maxx, miny, maxy, mint, maxt = dataset.bounds

    totalt = maxt - mint

    if not all(isinstance(x, tuple) for x in lengths):
        lengths = cast(Sequence[float], lengths)

        if not (isclose(sum(lengths), 1) or isclose(sum(lengths), totalt)):
            raise ValueError(
                "Sum of input lengths must equal 1 or the dataset's time length."
            )

        if any(n <= 0 for n in lengths):
            raise ValueError('All items in input lengths must be greater than 0.')

        if isclose(sum(lengths), 1):
            lengths = [totalt * f for f in lengths]

        lengths = [
            (mint + offset - length, mint + offset)  # type: ignore[operator]
            for offset, length in zip(accumulate(lengths), lengths)
        ]

    lengths = cast(Sequence[tuple[float, float]], lengths)

    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in lengths
    ]

    _totalt = 0.0
    for i, (start, end) in enumerate(lengths):
        if start >= end:
            raise ValueError(
                'Pairs of timestamps in lengths must have end greater than start.'
            )

        if start < mint or end > maxt:
            raise ValueError(
                "Pairs of timestamps in lengths can't be out of dataset's time bounds."
            )

        if any(start < x < end or start < y < end for x, y in lengths[i + 1 :]):
            raise ValueError("Pairs of timestamps in lengths can't overlap.")

        # Remove one microsecond from each BoundingBox's maxt to avoid overlapping
        offset = 0 if i == len(lengths) - 1 else 1e-6
        roi = BoundingBox(minx, maxx, miny, maxy, start, end - offset)
        j = 0
        for hit in dataset.index.intersection(tuple(roi), objects=True):
            box = BoundingBox(*hit.bounds)
            new_box = box & roi
            if new_box.volume > 0:
                new_indexes[i].insert(j, tuple(new_box), hit.object)
                j += 1

        _totalt += end - start

    if not isclose(_totalt, totalt):
        raise ValueError(
            "Pairs of timestamps in lengths must cover dataset's time bounds."
        )

    new_datasets = []
    for index in new_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        new_datasets.append(ds)

    return new_datasets

