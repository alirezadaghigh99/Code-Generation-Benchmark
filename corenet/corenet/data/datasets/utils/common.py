def select_samples_by_category(
    sample_category_labels: List[Any],
    random_seed: int,
    num_samples_per_category: Optional[int] = None,
    percentage_of_samples_per_category: Optional[float] = None,
) -> List[int]:
    """
    Randomly selects a specified number/percentage of samples from each category.

    Only one of `num_samples_per_category` and `percentage_of_samples_per_category` should be provided.
    Selects all the samples if neither of them are provided.

    Args:
        sample_category_labels: A list of category labels.
        random_seed: An integer seed to use for random selection.
        num_samples_per_category: An optional integer indicating the number of samples to select from each category.
        percentage_of_samples_per_category: An optional float in the range (0, 100] indicating the percentage of
            samples to select from each category.

    Returns:
        A list of (integer) indices of the selected samples.

    Raises:
        ValueError if both `num_samples_per_category` and `percentage_of_samples_per_category` are provided.
    """
    if (
        num_samples_per_category is not None
        and percentage_of_samples_per_category is not None
    ):
        raise ValueError(
            "Only one of `num_samples_per_category` and `percentage_of_samples_per_category` should be provided."
        )

    if num_samples_per_category is None and percentage_of_samples_per_category is None:
        return list(range(len(sample_category_labels)))

    if num_samples_per_category is not None and num_samples_per_category < 1:
        raise ValueError("`num_samples_per_category` should be greater than 0.")

    if percentage_of_samples_per_category is not None:
        if not 0 < percentage_of_samples_per_category <= 100:
            raise ValueError(
                "`percentage_of_samples_per_category` should be in the range (0, 100]."
            )

    category_specific_samples = collections.defaultdict(list)
    for ind, label in enumerate(sample_category_labels):
        category_specific_samples[label].append(ind)

    rng = random.Random(random_seed)
    selected_sample_indices = []
    for label, sample_indices in category_specific_samples.items():
        rng.shuffle(sample_indices)
        if num_samples_per_category:
            num_samples = num_samples_per_category
        else:
            num_samples = int(
                percentage_of_samples_per_category * len(sample_indices) / 100
            )
        num_samples = min(num_samples, len(sample_indices))
        selected_sample_indices += sample_indices[:num_samples]

    return selected_sample_indices

