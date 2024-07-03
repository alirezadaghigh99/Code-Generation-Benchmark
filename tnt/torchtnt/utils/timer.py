def get_synced_durations_histogram(
    recorded_durations: Dict[str, List[float]],
    percentiles: Sequence[float],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, float]]:
    """Synchronizes the recorded durations across ranks.

    Args:
        recorded_durations: The mapping of durations to sync and compute histograms from.
        percentiles: The percentiles to compute. Values should be in the range [0, 100].
        pg (optional): The process group to use for synchronization. Defaults to the global process group.

    Returns:
        A dictionary mapping the action names to a dictionary of the computed percentiles, along with the mean duration of each action.

    Raises:
        ValueError: If the input percentiles are not in the range [0, 100].
    """
    _validate_percentiles(percentiles)
    synced_durations = _sync_durations(recorded_durations, pg)
    return get_durations_histogram(synced_durations, percentiles=percentiles)