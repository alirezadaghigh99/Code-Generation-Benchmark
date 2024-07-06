def get_available_requested_snapshots(
    requested_snapshots: List[str],
    available_snapshots: List[str],
) -> List[str]:
    """
    Intersects the requested snapshot names with the available snapshots.

    Returns: snapshot names
    """
    snapshot_names = []
    missing_snapshots = []
    for snap in requested_snapshots:
        if snap in available_snapshots:
            snapshot_names.append(snap)
        else:
            missing_snapshots.append(snap)

    if len(snapshot_names) == 0:
        raise ValueError(
            f"None of the requested snapshots were found: \n{missing_snapshots}"
        )
    elif len(missing_snapshots) > 0:
        print(
            f"The following requested snapshots were not found and will be skipped:\n"
            f"{missing_snapshots}"
        )

    return snapshot_names

def get_snapshots_by_index(
    idx: Union[int, str], available_snapshots: List[str],
) -> List[str]:
    """
    Assume available_snapshots is ordered in ascending order. Returns snapshot names.
    """
    if (
        isinstance(idx, int)
        and -len(available_snapshots) <= idx < len(available_snapshots)
    ):
        return [available_snapshots[idx]]
    elif idx == "all":
        return available_snapshots

    raise IndexError(
        f"Invalid index: {idx}. The index should be an int less than the number of "
        f"available snapshots, negative indexing is supported. The keyword 'all' "
        f"is also a valid option."
    )

