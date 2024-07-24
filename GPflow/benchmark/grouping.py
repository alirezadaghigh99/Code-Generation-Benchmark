class GroupingSpec:
    by: Sequence[GroupingKey]
    """ Keys to group by. """

    minimise: bool
    """ Whether to ignore redundant keys. """

    def __post_init__(self) -> None:
        assert len(self.by) == len(set(self.by)), f"'by' must have unique values. Found {self.by}."

