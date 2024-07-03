def rand_split_train_val(
    datapipe: IterDataPipe,
    train_perc: float,
    random_seed: int = 0,
) -> Tuple[IterDataPipe, IterDataPipe]:
    """Via uniform random sampling, generates two IterDataPipe instances representing
    disjoint train and val splits of the given IterDataPipe.

    Args:
        datapipe (IterDataPipe): datapipe to split.
        train_perc (float): value in range (0.0, 1.0) specifying target proportion of
            datapipe samples to include in train split. Note that the actual proportion
            is not guaranteed to match train_perc exactly.
        random_seed (int): determines split membership for a given sample
            and train_perc. Use the same value across calls to generate consistent splits.
    Example::

        datapipe = criteo_terabyte(
            ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        )
        train_datapipe, val_datapipe = rand_split_train_val(datapipe, 0.75)
        train_batch = next(iter(train_datapipe))
        val_batch = next(iter(val_datapipe))
    """
    if not 0.0 < train_perc < 1.0:
        raise ValueError("train_perc must be in range (0.0, 1.0)")

    return _RandFilter(
        datapipe, partial(_rand_train_filter_fn, train_perc), random.Random(random_seed)
    ), _RandFilter(
        datapipe, partial(_rand_val_filter_fn, train_perc), random.Random(random_seed)
    )class ParallelReadConcat(IterDataPipe):
    r""":class:`ParallelReadConcat`.

    Iterable DataPipe that concatenates multiple Iterable DataPipes.
    When used with a DataLoader, assigns a subset of datapipes to each DataLoader worker
    to allow for parallel reading.
    Args:
        datapipes: IterDataPipe instances to read from.
        dp_selector: function that each DataLoader worker would use to determine the subset of datapipes
        to read from.
    Example::

        datapipes = [
            criteo_terabyte(
                (f"/home/local/datasets/criteo/shard_{idx}.tsv",),
            )
            .batch(100)
            .collate()
            for idx in range(4)
        ]
        dataloader = DataLoader(
            ParallelReadConcat(*datapipes), num_workers=4, batch_size=None
        )
    """

    def __init__(
        self,
        *datapipes: IterDataPipe,
        dp_selector: Callable[
            [Sequence[IterDataPipe]], Sequence[IterDataPipe]
        ] = _default_dp_selector,
    ) -> None:
        super().__init__()
        self.datapipes: Tuple[IterDataPipe, ...] = datapipes
        self.dp_selector = dp_selector

    # pyre-ignore[3]
    def __iter__(self) -> Iterator[Any]:
        selected_dps = self.dp_selector(self.datapipes)
        for dp in selected_dps:
            for data in dp:
                yield datadef idx_split_train_val(
    datapipe: IterDataPipe,
    train_perc: float,
    decimal_places: int = 2,
    key_fn: Callable[[int], int] = _default_key_fn,
) -> Tuple[IterDataPipe, IterDataPipe]:
    if not 0.0 < train_perc < 1.0:
        raise ValueError("train_perc must be in range (0.0, 1.0)")
    return (
        _IdxFilter(datapipe, partial(train_filter, key_fn, train_perc, decimal_places)),
        _IdxFilter(datapipe, partial(val_filter, key_fn, train_perc, decimal_places)),
    )