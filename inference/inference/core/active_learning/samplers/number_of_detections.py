def is_in_range(
    value: int,
    more_than: Optional[int],
    less_than: Optional[int],
) -> bool:
    # calculates value > more_than and value < less_than, with optional borders of range
    less_than_satisfied, more_than_satisfied = less_than is None, more_than is None
    if less_than is not None and value < less_than:
        less_than_satisfied = True
    if more_than is not None and value > more_than:
        more_than_satisfied = True
    return less_than_satisfied and more_than_satisfied